"""ASR backends. Interface: `ASRBackend.transcribe(audio_path) -> str`."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from .cache import content_hash
from .tts import decode_mock_audio


class ASRBackend(Protocol):
    name: str

    def cache_params(self) -> dict: ...

    def transcribe(self, audio_path: str | Path) -> str: ...


class MockASR:
    """Decodes MockTTS audio and injects deterministic, accent-dependent homophone swaps.

    This is a SIMULATION for tests and offline UI demos – its numbers mean nothing.
    """

    name = "mock"
    BASE_RATE = {"de": 0.05, "tr": 0.30, "ru": 0.25, "ar": 0.35, "en": 0.20, "none": 0.10}

    def __init__(self, partners: dict[str, list[str]] | None = None, seed: int = 0):
        self.partners = {k.lower(): v for k, v in (partners or {}).items()}  # word -> homophones
        self.seed = seed

    def cache_params(self) -> dict:
        return {"backend": "mock-asr-v1", "seed": self.seed,
                "lex": content_hash(sorted((k, sorted(v)) for k, v in self.partners.items()))}

    def transcribe(self, audio_path):
        req = decode_mock_audio(audio_path)
        if req is None:
            return ""
        rate = self.BASE_RATE.get(req["accent"], 0.15) * (0.4 + 1.2 * float(req["cfg"]))
        out = []
        for i, tok in enumerate(req["text"].split()):
            core = tok.strip(".,!?;:\"'()")
            rng = random.Random(content_hash([self.seed, req["text"], req["accent"], req["cfg"], i]))
            alts = self.partners.get(core.lower())
            if alts and rng.random() < rate:
                tok = tok.replace(core, sorted(alts)[rng.randrange(len(alts))])
            elif rng.random() < rate / 6:
                continue  # simulated deletion
            out.append(tok)
        return " ".join(out)


class FasterWhisperASR:
    """faster-whisper with fixed decoding settings (beam size, temperature) for reproducibility."""

    name = "faster_whisper"

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8",
                 language: str = "de", beam_size: int = 5, temperature: float = 0.0,
                 condition_on_previous_text: bool = False, vad_filter: bool = False):
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.kw = dict(language=language, beam_size=beam_size, temperature=temperature,
                       condition_on_previous_text=condition_on_previous_text, vad_filter=vad_filter)
        self._model = None

    def cache_params(self) -> dict:
        return {"backend": "faster-whisper", "model": self.model_name, "compute_type": self.compute_type,
                **self.kw}

    def load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path):
        segments, _ = self.load().transcribe(str(audio_path), **self.kw)
        return " ".join(s.text.strip() for s in segments).strip()


def make_asr(cfg: dict, partners: dict[str, list[str]] | None = None) -> ASRBackend:
    if cfg["backends"]["asr"] == "mock":
        return MockASR(partners, cfg.get("seed", 0))
    a = cfg["asr"]
    return FasterWhisperASR(a.get("model", "small"), a.get("device", "cpu"), a.get("compute_type", "int8"),
                            a.get("language", "de"), a.get("beam_size", 5), a.get("temperature", 0.0),
                            a.get("condition_on_previous_text", False), a.get("vad_filter", False))
