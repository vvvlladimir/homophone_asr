from homophone_asr.analyze import (CORRECT, DELETED, EXACT, HOMOPHONE, MISSING, OTHER, WITHIN_GROUP,
                                   aggregate, align, analyze_sentence, analyze_word,
                                   confusion_directions, diff_spans, find_target, normalize)

G = ["mehr", "Meer"]


def test_normalize_keeps_umlauts_and_eszett():
    assert normalize("  Die STRASSE, groß! Äpfel?„") == "die strasse groß äpfel"
    assert normalize("Gross") != normalize("groß")
    assert normalize("") == "" and normalize(None) == ""
    assert normalize("über") == "über"  # NFD -> NFC


def test_align_counts():
    ops, c = align("Das Meer ist tief.", "das mehr ist sehr tief")
    assert (c["subs"], c["ins"], c["dels"], c["n_ref"]) == (1, 1, 0, 4)
    assert abs(c["wer"] - 0.5) < 1e-9
    assert [o.op for o in ops].count("equal") == 3


def test_align_empty_edges():
    assert align("", "")[1]["wer"] == 0.0
    assert align("", "hallo")[1]["wer"] == 1.0
    ops, c = align("a b", "")
    assert c["dels"] == 2 and c["wer"] == 1.0


def test_sentence_homophone_confusion():
    r = analyze_sentence("Das Meer ist tief.", "Das mehr ist tief.", "Meer", G)
    assert r["status"] == HOMOPHONE and r["recognized"] == "mehr" and r["target_idx"] == 1


def test_sentence_correct_case_insensitive():
    r = analyze_sentence("Ich will mehr Wasser.", "ich will MEHR wasser", "mehr", G)
    assert r["status"] == CORRECT and r["wer"] == 0


def test_sentence_other_and_deleted():
    assert analyze_sentence("Das Meer ist tief.", "Das Bier ist tief.", "Meer", G)["status"] == OTHER
    assert analyze_sentence("Das Meer ist tief.", "Das ist tief.", "Meer", G)["status"] == DELETED
    assert analyze_sentence("Das Meer ist tief.", "", "Meer", G)["status"] == DELETED
    assert analyze_sentence("Das Haus ist tief.", "x", "Meer", G)["status"] == MISSING


def test_target_offset_with_preceding_insertion():
    # insertion BEFORE the target must not shift the target index (off-by-one guard)
    r = analyze_sentence("Das Meer ist tief.", "Also das mehr ist tief.", "Meer", G)
    assert r["status"] == HOMOPHONE and r["recognized"] == "mehr"


def test_find_target_occurrence():
    assert find_target("Wir wollen mehr und noch mehr.", "mehr", 1) == 5
    assert find_target("Wir wollen mehr.", "mehr", 1) is None


def test_word_level_accepts_group():
    assert analyze_word("Meer", "Meer.", G)["status"] == EXACT
    w = analyze_word("Meer", "mehr", G)
    assert w["status"] == WITHIN_GROUP and w["accepted"]
    assert not analyze_word("Meer", "Bier", G)["accepted"]
    assert not analyze_word("Meer", "", G)["accepted"]
    assert not analyze_word("Meer", "mehr mehr", G)["accepted"]


def test_diff_spans_labels():
    r = analyze_sentence("Das Meer ist tief.", "das mehr ist sehr tief", "Meer", G)
    spans = diff_spans(r["ops"], r["target_idx"], r["status"], r["recognized"])
    labels = [l for _, l in spans]
    assert "homophone" in labels and "ins" in labels


def test_aggregate_and_directions():
    rows = []
    for acc, hyp in [("tr", "Das mehr ist tief."), ("tr", "Das Meer ist tief."), ("de", "Das Meer ist tief.")]:
        r = analyze_sentence("Das Meer ist tief.", hyp, "Meer", G)
        r.update(accent=acc, cfg_weight=0.5)
        rows.append(r)
    agg = aggregate(rows).set_index("accent")
    assert agg.loc["tr", "homophone_confusion_rate"] == 0.5 and agg.loc["de", "wer"] == 0
    assert agg.loc["tr", "wer"] == 0.125
    assert confusion_directions(rows) == {"meer → mehr": 1}
    assert aggregate([]).empty


def test_diff_spans_keep_word_separators():
    r = analyze_sentence("Die Katze schläft.", "Die Katze schläft.", "x", ["x"])
    assert diff_spans(r["ops"]) == [("die katze schläft", None)]
