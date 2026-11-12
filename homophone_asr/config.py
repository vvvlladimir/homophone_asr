"""Config loading. One YAML file; relative paths are resolved against the file's directory."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load YAML config, apply nested `overrides`, resolve paths to absolute."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    validate(cfg)
    root = path.resolve().parent
    cfg["_root"] = str(root)
    for k, v in (cfg.get("paths") or {}).items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else root / p)
    return cfg


ALLOWED = {"tts": {"chatterbox", "mock"}, "asr": {"faster_whisper", "mock"}, "lm": {"gbert", "mock"},
           "g2p": {"espeak", "mock"}}


def validate(cfg: dict) -> None:
    for k, ok in ALLOWED.items():
        v = (cfg.get("backends") or {}).get(k)
        if v not in ok:
            raise ValueError(f"config backends.{k}={v!r}; allowed: {sorted(ok)}")
    cw = (cfg.get("tts") or {}).get("cfg_weight", 0.5)
    if not 0.0 <= float(cw) <= 1.0:
        raise ValueError(f"tts.cfg_weight={cw} outside [0, 1]")


def mock_overrides() -> dict:
    """Overrides that switch every backend to its Mock implementation."""
    return {"backends": {"tts": "mock", "asr": "mock", "lm": "mock", "g2p": "mock"}}
