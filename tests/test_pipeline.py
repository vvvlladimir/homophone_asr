import json

import pytest

from homophone_asr import cli
from homophone_asr.analyze import CORRECT, HOMOPHONE
from homophone_asr.pipeline import Pipeline, save_results


@pytest.fixture
def pipe(cfg):
    return Pipeline(cfg, allow_download=False)


def test_mock_backends_selected(pipe):
    assert pipe.is_mock and pipe.tts.name == "mock" and pipe.asr.name == "mock"


def test_missing_ref_warns_not_blocks(pipe):
    r = pipe.run_sentence("Das Meer ist heute sehr ruhig.", "tr", 0.5, "Meer", "mehr|Meer")
    assert r["ref_used"] is False and "No accent reference" in r["warning"]
    assert r["status"] in (CORRECT, HOMOPHONE, "other_substitution", "deleted")


def test_cache_hit_and_reproducible(pipe):
    a = pipe.run_sentence("Ich möchte bitte mehr Wasser.", "ar", 0.5)
    b = pipe.run_sentence("Ich möchte bitte mehr Wasser.", "ar", 0.5)
    assert not a["tts_cached"] and b["tts_cached"] and b["asr_cached"]
    assert a["hyp"] == b["hyp"] and a["audio_path"] == b["audio_path"]
    c = pipe.run_sentence("Ich möchte bitte mehr Wasser.", "ar", 0.3)
    assert c["audio_path"] != a["audio_path"]          # cfg weight is part of the key


def test_cache_key_depends_on_ref_content(cfg, tmp_path):
    import numpy as np
    import soundfile as sf
    d = tmp_path / "refs" / "tr"
    d.mkdir(parents=True)
    sf.write(d / "a.wav", np.zeros(16000 * 6, dtype="float32"), 16000)
    p1 = Pipeline(cfg, allow_download=False)
    a = p1.synthesize("Hallo Welt.", "tr", 0.5)
    assert a["ref_used"]
    sf.write(d / "a.wav", np.ones(16000 * 6, dtype="float32") * 0.1, 16000)
    p2 = Pipeline(cfg, allow_download=False)
    assert p2.synthesize("Hallo Welt.", "tr", 0.5)["audio_path"] != a["audio_path"]


def test_experiment_and_save(pipe, tmp_path):
    rows = pipe.select(limit_pairs=2)
    assert len({r["pair_id"] for r in rows}) == 2
    res = pipe.run_experiment(["de", "ar"], [0.0, 0.5], rows, mode="both")
    assert len(res["sentences"]) == len(rows) * 4 and len(res["words"]) == 4 * 4
    s = res["summary"]
    assert set(s["accent"]) == {"de", "ar"} and s["wer"].between(0, 5).all()
    res2 = pipe.run_experiment(["de", "ar"], [0.0, 0.5], rows, mode="both")
    assert res2["summary"].equals(s)                    # same input -> same output
    d = save_results(res, tmp_path, "x")
    assert "MOCK" in (d / "report.md").read_text()
    assert json.loads((d / "rows.json").read_text())["meta"]["mock"] is True


def test_free_text_without_homophone(pipe):
    r = pipe.run_sentence("Guten Morgen zusammen.", "de", 0.0)
    assert r["target"] is None and r["status"] is None and r["wer"] >= 0


def test_cli_run_mock(tmp_path, capsys, monkeypatch):
    import yaml
    from homophone_asr.config import DEFAULT_CONFIG
    c = yaml.safe_load(DEFAULT_CONFIG.read_text())
    c["paths"].update(cache_dir=str(tmp_path / "c"), results_dir=str(tmp_path / "r"),
                      refs_dir=str(tmp_path / "refs"),
                      sentences=str(DEFAULT_CONFIG.parent / "data/sentences_de.csv"),
                      lexicon=str(DEFAULT_CONFIG.parent / "data/homophone_words_de.txt"))
    c["tts"]["common_voice"]["enabled"] = False
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    rc = cli.main(["--config", str(p), "--mock", "run", "--accents", "tr,ru", "--cfg", "0,0.5",
                   "--limit-pairs", "1", "--name", "t"])
    assert rc == 0 and (tmp_path / "r" / "t" / "summary.csv").exists()
    assert "MOCK" in capsys.readouterr().out


def test_cli_rejects_bad_cfg():
    with pytest.raises(SystemExit):
        cli.main(["--mock", "run", "--cfg", "0,2"])


def test_no_ref_accents_do_not_share_cache(pipe):
    a = pipe.synthesize("Hallo Welt.", "tr", 0.5)
    b = pipe.synthesize("Hallo Welt.", "ru", 0.5)
    assert a["audio_path"] != b["audio_path"]


def test_cli_validation_errors(capsys):
    for argv in (["--mock", "try", "x", "--cfg", "1.5"], ["--mock", "run", "--accents", ""],
                 ["--mock", "run", "--cfg", ""], ["--mock", "run", "--limit-pairs", "0"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    assert cli.main(["--config", "/nonexistent.yaml", "groups"]) == 2


def test_bad_backend_rejected(cfg):
    from homophone_asr.config import validate
    cfg["backends"]["tts"] = "mok"
    with pytest.raises(ValueError):
        validate(cfg)


def test_dataset_sentence_uses_declared_target(pipe):
    r = pipe.run_sentence("Das Meer ist heute sehr ruhig.", "de", 0.0)
    assert r["target"] == "Meer" and r["group"] == "mehr|Meer"
    free = pipe.run_sentence("Das Lied und das Meer.", "de", 0.0)
    assert free["target"] in ("das", "Lied", "Meer")


def test_read_as_accent_changes_audio_and_keeps_default_key(pipe):
    a = pipe.synthesize("Das Meer ist tief.", "tr", 0.5)
    b = pipe.synthesize("Das Meer ist tief.", "tr", 0.5, read_as="accent")
    c = pipe.synthesize("Das Meer ist tief.", "tr", 0.5, read_as="de")
    assert a["read_as"] == "de" and b["read_as"] == "tr"
    assert a["audio_path"] == c["audio_path"] != b["audio_path"]
    # native German / no-reference voices fall back to German rules
    assert pipe.text_language("de", "accent") == "de" and pipe.text_language("none", "accent") == "de"
    with pytest.raises(ValueError):
        pipe.text_language("tr", "xx")


def test_experiment_read_as_in_meta(pipe):
    res = pipe.run_experiment(["tr"], [1.0], pipe.select(limit_pairs=1), read_as="accent")
    assert res["meta"]["read_as"] == "accent" and all(r["read_as"] == "tr" for r in res["sentences"])


def test_cli_try_read_as(capsys):
    assert cli.main(["--mock", "try", "Das Meer ist tief.", "--accent", "ru", "--cfg", "1.0",
                     "--read-as", "accent"]) == 0
    assert '"read_as": "ru"' in capsys.readouterr().out
