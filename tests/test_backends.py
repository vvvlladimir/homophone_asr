from homophone_asr.asr import MockASR
from homophone_asr.cache import Cache, content_hash
from homophone_asr.detector import MockLM
from homophone_asr.tts import MockTTS, decode_mock_audio


def test_config_mock(cfg):
    assert cfg["backends"]["tts"] == "mock" and cfg["asr"]["beam_size"] == 5


def test_mock_tts_roundtrip(tmp_path):
    p = tmp_path / "a.wav"
    MockTTS().synthesize("Das Meer ist groß.", p, "tr", None, 0.3)
    req = decode_mock_audio(p)
    assert req["text"] == "Das Meer ist groß." and req["accent"] == "tr" and req["cfg"] == 0.3


def test_mock_asr_deterministic_and_native_clean(tmp_path):
    p = tmp_path / "a.wav"
    MockTTS().synthesize("Ich will mehr Wasser.", p, "ar", None, 0.5)
    asr = MockASR({"mehr": ["Meer"], "meer": ["mehr"]})
    assert asr.transcribe(p) == asr.transcribe(p)
    q = tmp_path / "b.wav"
    MockTTS().synthesize("Ich will mehr Wasser.", q, "de", None, 0.0)
    assert MockASR({}).transcribe(q) == "Ich will mehr Wasser."


def test_mock_lm_prefers_context():
    lm = MockLM(["Das Meer ist tief und blau.", "Ich will mehr Wasser trinken."])
    text = "Das mehr ist tief."
    s = lm.candidate_logprobs(text, (4, 8), ["mehr", "Meer"])
    assert s["Meer"] > s["mehr"]


def test_cache(tmp_path):
    c = Cache(tmp_path)
    k = content_hash({"a": 1, "b": [1, 2]})
    assert k == content_hash({"b": [1, 2], "a": 1})
    assert c.get_json("x", k) is None
    c.put_json("x", k, {"v": "ä"})
    assert c.get_json("x", k) == {"v": "ä"}
