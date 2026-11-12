import shutil

import pytest

from homophone_asr.confusion import (EspeakG2P, Lexicon, MockG2P, build_groups, check_declared_groups,
                                     load_lexicon, read_sentences, read_words)

HAS_ESPEAK = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def test_mock_groups_basic():
    groups = build_groups(["mehr", "Meer", "Ende", "Ente", "Rad", "Rat", "Weise", "weise"], MockG2P())
    words = [g.words for g in groups]
    assert ["Meer", "mehr"] in words and ["Rad", "Rat"] in words
    assert not any("Ende" in w for w in words)          # Ende/Ente are NOT homophones
    assert not any("Weise" in w or "weise" in w for w in words)  # case variants = one spelling


def test_lexicon_lookup_case_insensitive():
    lex = Lexicon(build_groups(["mehr", "Meer", "Stadt", "statt"], MockG2P()))
    assert lex.partners("MEER") == ["mehr"] and lex.partners("stadt") == ["statt"]
    assert "Haus" not in lex and lex.partners("Haus") == []


def test_build_groups_empty_and_blank():
    assert build_groups([], MockG2P()) == []
    assert build_groups(["", "  "], MockG2P()) == []


def test_dataset_wellformed(cfg):
    rows = read_sentences(cfg["paths"]["sentences"])
    assert len(rows) >= 20
    by_target = {}
    for r in rows:
        assert r["target"].lower() in r["sentence"].lower()
        by_target.setdefault(r["target"], 0)
        by_target[r["target"]] += 1
    assert all(n >= 2 for n in by_target.values())        # 2 contexts per homophone


def test_dataset_groups_mock(cfg):
    rows = read_sentences(cfg["paths"]["sentences"])
    assert check_declared_groups(rows, load_lexicon(cfg, rows)) == []


@pytest.mark.skipif(not HAS_ESPEAK, reason="espeak-ng not installed")
def test_dataset_groups_espeak(cfg):
    cfg["backends"]["g2p"] = "espeak"
    rows = read_sentences(cfg["paths"]["sentences"])
    lex = load_lexicon(cfg, rows)
    assert check_declared_groups(rows, lex) == []
    assert lex.partners("das") == ["dass"]
    assert lex.group_of("Ende") is None or "Ente" not in lex.group_of("Ende").words


@pytest.mark.skipif(not HAS_ESPEAK, reason="espeak-ng not installed")
def test_espeak_umlaut_and_eszett():
    g = EspeakG2P()
    a, b, c = g.phonemize(["Häute", "heute", "Fuß"])
    assert a == b and c and "ˈ" not in c


def test_read_words_ignores_comments(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("# c\nmehr\n\nMeer # x\n", encoding="utf-8")
    assert read_words(p) == ["mehr", "Meer"]
