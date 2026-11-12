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
MOCK_SR = 16000


class TTSBackend(Protocol):
    name: str

    def cache_params(self) -> dict:
        """Everything (besides text/ref/cfg) that changes the audio; used for the cache key."""
        ...

    def synthesize(self, text: str, out_path: Path, accent: str, ref_path: str | None,
                   cfg_weight: float) -> None:
        ...


class MockTTS:
    """Writes a wav whose samples encode the request as JSON (decoded by MockASR). No model needed."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def cache_params(self) -> dict:
        return {"backend": "mock-tts-v1", "seed": self.seed}

    def synthesize(self, text, out_path, accent, ref_path, cfg_weight):
        payload = json.dumps({"text": text, "accent": accent, "cfg": round(float(cfg_weight), 3),
                              "ref": bool(ref_path)}, ensure_ascii=False).encode("utf-8")
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
