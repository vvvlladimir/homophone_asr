import os

import pytest

from homophone_asr.confusion import Lexicon, MockG2P, build_groups
from homophone_asr.detector import (HomophoneDetector, MockLM, evaluate_detector, swap_target,
                                    _is_sentence_start)

LEX = Lexicon(build_groups(["mehr", "Meer", "Rad", "Rat", "das", "dass"], MockG2P()))
CORPUS = ["Das Meer ist heute sehr ruhig.", "Wir fahren im Sommer ans Meer.",
          "Ich möchte bitte mehr Wasser.", "Ich fahre jeden Tag mit dem Rad.",
          "Danke für deinen guten Rat."]


class FixedLM:
    """LM stub returning preset scores (tests threshold logic exactly)."""
    name = "fixed"

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def candidate_logprobs(self, text, span, candidates):
        self.calls.append(candidates)
        return {c: self.scores.get(c, -50.0) for c in candidates}


def test_flags_wrong_homophone_mock():
    det = HomophoneDetector(MockLM(CORPUS), LEX, threshold=1.0)
    f = [x for x in det.detect("Das mehr ist heute sehr ruhig.") if x.word == "mehr"][0]
    assert f.flagged and f.best == "Meer" and f.margin > 1.0
    g = [x for x in det.detect("Ich möchte bitte mehr Wasser.") if x.word == "mehr"][0]
    assert not g.flagged


def test_threshold_boundary():
    det = HomophoneDetector(FixedLM({"Meer": -1.0, "mehr": -3.0}), LEX, threshold=2.0)
    assert det.detect("Ich will Meer.")[0].flagged is False            # margin -2
    f = det.detect("Ich will mehr.")[0]
    assert f.margin == pytest.approx(2.0) and f.flagged                # margin == threshold -> flag
    det.threshold = 2.01
    assert not det.detect("Ich will mehr.")[0].flagged


def test_probs_sum_to_one_and_no_finding_without_homophones():
    det = HomophoneDetector(MockLM(CORPUS), LEX)
    f = det.detect("Wir wollen mehr.")[0]
    assert sum(f.probs.values()) == pytest.approx(1.0)
    assert det.detect("Guten Morgen, Welt!") == [] and det.detect("") == []


def test_sentence_start_capitalization():
    lm = FixedLM({})
    det = HomophoneDetector(lm, LEX)
    det.detect("Mehr Wasser bitte.")
    assert "Meer" in lm.calls[0] and "Mehr" in lm.calls[0]
    lm.calls.clear()
    det.detect("Wir sehen das Meer.")
    assert lm.calls[-1] == ["Meer", "mehr"]                            # canonical case mid-sentence
    assert _is_sentence_start("Hallo. Meer", 7) and not _is_sentence_start("Das Meer", 4)


def test_original_case_variant_counts_as_original():
    # ASR wrote 'meer' lowercase: original scored as best of {'meer','Meer'} -> not falsely flagged
    det = HomophoneDetector(FixedLM({"meer": -9.0, "Meer": -1.0, "mehr": -2.5}), LEX)
    f = det.detect("Wir sehen das meer.")[-1]
    assert f.scores["meer"] == -1.0 and not f.flagged


def test_highlight_roundtrip():
    det = HomophoneDetector(FixedLM({"Meer": -1.0, "mehr": -9.0}), LEX)
    text = "Ich will mehr Wasser."
    spans = det.highlight(text, det.detect(text))
    assert "".join(s for s, _ in spans) == text
    assert any(l and l.startswith("suspicious → Meer") for _, l in spans)


def test_swap_target():
    assert swap_target("Mehr geht nicht.", "mehr", "Meer") == "Meer geht nicht."
    assert swap_target("Wir wollen mehr.", "mehr", "Meer") == "Wir wollen Meer."
    assert swap_target("Nichts hier.", "mehr", "Meer") is None


def test_evaluate_detector_mock():
    rows = [{"id": "1", "target": "Meer", "sentence": "Das Meer ist heute sehr ruhig."}]
    e = evaluate_detector(HomophoneDetector(MockLM(CORPUS), LEX, 1.0), rows)
    assert e["n_swapped"] == 1 and e["detection_rate"] == 1.0 and e["false_alarm_rate"] == 0.0


@pytest.mark.skipif(os.environ.get("HOMOPHONE_REAL") != "1", reason="set HOMOPHONE_REAL=1 (downloads gbert)")
def test_gbert_real():
    from homophone_asr.detector import HFMaskedLM
    det = HomophoneDetector(HFMaskedLM(), LEX, 2.0)
    f = [x for x in det.detect("Ich will Meer Wasser.") if x.word == "Meer"][0]
    assert f.flagged and f.best == "mehr"


def test_allcaps_eszett_stays_distinct():
    lex = Lexicon(build_groups(["Maß", "Mass"], MockG2P()))
    if not lex.groups:  # MockG2P does not merge these; build the group by hand
        from homophone_asr.confusion import HomophoneGroup
        lex = Lexicon([HomophoneGroup("mas", ["Maß", "Mass"])])
    lm = FixedLM({})
    HomophoneDetector(lm, lex).detect("EIN MASS BIER")
    assert len(set(lm.calls[0])) == 2 and "MAẞ" in lm.calls[0]


def test_closing_quote_and_semicolon_not_sentence_start():
    assert not _is_sentence_start('er sagte "ja" das', 14)
    assert not _is_sentence_start("ich sehe; das", 10)
    assert _is_sentence_start('Er sagte: "Das', 11)


def test_evaluate_counts_originals_once():
    lex = Lexicon(build_groups(["mehr", "Meer", "Rad", "Rat"], MockG2P()))
    rows = [{"id": "1", "target": "Meer", "sentence": "Das Meer ist heute sehr ruhig."},
            {"id": "2", "target": "Haus", "sentence": "Das Haus."}]
    e = evaluate_detector(HomophoneDetector(MockLM(CORPUS), lex, 1.0), rows)
    assert e["n_original"] == 1 and e["skipped"] == 1
