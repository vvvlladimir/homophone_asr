"""Content-hash cache. Keys are SHA-256 of a canonical JSON of all inputs that affect the output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


class Cache:
    """Directory cache: <root>/<namespace>/<key>.<ext>."""

    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def path(self, namespace: str, key: str, ext: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.{ext}"

    def get_json(self, namespace: str, key: str) -> Any | None:
        p = self.path(namespace, key, "json")
        if self.enabled and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None  # corrupt entry -> recompute
        return None

    def put_json(self, namespace: str, key: str, value: Any) -> None:
        p = self.path(namespace, key, "json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
