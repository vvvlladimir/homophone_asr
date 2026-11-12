"""TTS backends. Interface: `TTSBackend.synthesize(text, out_path, accent, ref_path, cfg_weight)`.

Accent trick (Chatterbox Multilingual): German text + a reference clip in ANOTHER language makes the
output inherit that language's accent; `cfg_weight` controls how strongly (0.0 suppresses it).
These accents are SYNTHETIC and NOT validated against real L2 speech.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

MOCK_MAGIC = [12345, -12345, 4242, -4242]
# languages Chatterbox Multilingual 0.1.7 accepts as language_id
CHATTERBOX_LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
                    "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}
MOCK_SR = 16000


class TTSBackend(Protocol):
    name: str

    def cache_params(self) -> dict:
        """Everything (besides text/ref/cfg) that changes the audio; used for the cache key."""
        ...

    def synthesize(self, text: str, out_path: Path, accent: str, ref_path: str | None,
                   cfg_weight: float, language_id: str | None = None) -> None:
        """language_id: pronunciation rules used to read the (German) text; None = backend default."""
        ...


class MockTTS:
    """Writes a wav whose samples encode the request as JSON (decoded by MockASR). No model needed."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def cache_params(self) -> dict:
        return {"backend": "mock-tts-v1", "seed": self.seed}

    def synthesize(self, text, out_path, accent, ref_path, cfg_weight, language_id=None):
        payload = json.dumps({"text": text, "accent": accent, "cfg": round(float(cfg_weight), 3),
                              "ref": bool(ref_path), "lang": language_id or "de"},
                             ensure_ascii=False).encode("utf-8")
        data = np.array(MOCK_MAGIC + [len(payload)] + list(payload), dtype=np.int16)
        t = np.arange(int(0.25 * MOCK_SR)) / MOCK_SR
        tone = (0.05 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        sf.write(str(out_path), np.concatenate([data, tone]), MOCK_SR, subtype="PCM_16")


def decode_mock_audio(path: str | Path) -> dict | None:
    """Inverse of MockTTS.synthesize; returns None for non-mock audio."""
    data, _ = sf.read(str(path), dtype="int16")
    if data.ndim > 1 or len(data) < 5 or list(data[:4]) != MOCK_MAGIC:
        return None
    n = int(data[4])
    return json.loads(bytes(int(x) for x in data[5:5 + n]).decode("utf-8"))


class ChatterboxTTS:
    """Chatterbox Multilingual (MIT). German text; accent from a foreign-language reference clip."""

    name = "chatterbox"

    def __init__(self, device: str = "cpu", language_id: str = "de", exaggeration: float = 0.5,
                 temperature: float = 0.8, seed: int = 1234):
        self.device, self.language_id = device, language_id
        self.exaggeration, self.temperature, self.seed = exaggeration, temperature, seed
        self._model = None
        self._builtin = None
        self._conds: dict[str, object] = {}

    def cache_params(self) -> dict:
        try:
            from importlib.metadata import version
            v = version("chatterbox-tts")
        except Exception:
            v = "unknown"
        return {"backend": "chatterbox-mtl", "version": v, "lang": self.language_id,
                "exaggeration": self.exaggeration, "temperature": self.temperature, "seed": self.seed}

    def load(self):
        if self._model is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            torch.set_num_threads(max(1, torch.get_num_threads()))
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self._builtin = self._model.conds  # prepare_conditionals() replaces, never mutates
        return self._model

    def synthesize(self, text, out_path, accent, ref_path, cfg_weight, language_id=None):
        import random

        import torch
        m = self.load()
        if ref_path:
            if ref_path not in self._conds:
                m.prepare_conditionals(ref_path, exaggeration=self.exaggeration)
                self._conds[ref_path] = m.conds
            m.conds = self._conds[ref_path]
        else:
            m.conds = self._builtin
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        wav = m.generate(text, language_id=language_id or self.language_id, exaggeration=self.exaggeration,
                         cfg_weight=float(cfg_weight), temperature=self.temperature)
        sf.write(str(out_path), wav.squeeze(0).cpu().numpy(), m.sr, subtype="PCM_16")


def make_tts(cfg: dict) -> TTSBackend:
    if cfg["backends"]["tts"] == "mock":
        return MockTTS(cfg.get("seed", 0))
    t = cfg["tts"]
    return ChatterboxTTS(t.get("device", "cpu"), t.get("language_id", "de"), t.get("exaggeration", 0.5),
                         t.get("temperature", 0.8), cfg.get("seed", 1234))
