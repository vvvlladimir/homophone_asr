"""Reference voices for the accent trick: refs/<lang>/*.wav (5-10 s).

Missing refs are fetched from Mozilla Common Voice 17.0 (CC0) via a public parquet mirror on the
Hugging Face Hub. If that fails, callers fall back to the TTS default voice WITH a warning.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)
REF_SR = 24000
# optional metadata preferences per language (Common Voice 'accent' field), purely a soft filter
ACCENT_HINT = {"en": "United States English", "de": "Deutschland"}


def _duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:  # unreadable file
        return -1.0


def find_refs(refs_dir: str | Path, lang: str, min_s: float = 5.0, max_s: float = 10.0) -> list[Path]:
    d = Path(refs_dir) / lang
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("*.wav")) if min_s <= _duration(p) <= max_s]


def fetch_common_voice(lang: str, refs_dir: str | Path, repo: str = "fixie-ai/common_voice_17_0",
                       split: str = "test", n: int = 1, min_s: float = 5.0, max_s: float = 10.0,
                       prefer_gender: str | None = "male_masculine") -> list[Path]:
    """Download n clean 5-10 s clips for `lang` (reads only the first parquet row groups)."""
    import librosa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{repo}/{lang}"
    files = sorted(fs.glob(f"{base}/{split}/**/*.parquet")) or sorted(fs.glob(f"{base}/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Common Voice parquet for {lang!r}/{split} in {repo}")
    pf = pq.ParquetFile(fs.open(files[0]))
    cands = []
    for rg in range(min(pf.num_row_groups, 6)):
        t = pf.read_row_group(rg, columns=["path", "audio", "sentence", "up_votes", "down_votes",
                                           "gender", "accent"]).to_pylist()
        for r in t:
            if r["down_votes"] or not r["audio"] or not r["audio"].get("bytes"):
                continue
            try:
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            except Exception:
                continue
            dur = len(wav) / sr
            if min_s <= dur <= max_s:
                acc, hint = (r["accent"] or "").strip(), ACCENT_HINT.get(lang)
                # prefer plain/unmarked speakers: penalize any self-described accent/impairment tag
                score = min(r["up_votes"] or 0, 4) + 3 * (prefer_gender is not None and r["gender"] == prefer_gender) \
                    + (5 if hint and acc.startswith(hint) and "," not in acc else 0) \
                    - (5 if acc and not (hint and acc.startswith(hint) and "," not in acc) else 0)
                cands.append((-score, r["path"], wav.mean(axis=1), sr, r))
        if len(cands) >= 20 * n:
            break
    if not cands:
        raise RuntimeError(f"no 5-10 s clips found for {lang!r}")
    cands.sort(key=lambda c: (c[0], c[1]))
    out_dir = Path(refs_dir) / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, meta = [], []
    for _, name, wav, sr, r in cands[:n]:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=REF_SR) if sr != REF_SR else wav
        wav = 0.9 * wav / max(1e-6, float(np.abs(wav).max()))
        p = out_dir / f"cv17_{Path(name).stem}.wav"
        sf.write(str(p), wav, REF_SR, subtype="PCM_16")
        paths.append(p)
        meta.append(f"- `{p.name}`: Common Voice 17.0 ({lang}), clip `{Path(name).name}`, gender={r['gender'] or 'n/a'}, "
                    f"accent field={r['accent'] or 'n/a'}, transcript: \"{r['sentence']}\"")
    (out_dir / "SOURCE.md").write_text(
        f"# Reference clips for `{lang}`\n\nSource: Mozilla Common Voice 17.0 via HF mirror `{repo}`, "
        "license CC0-1.0. Do not attempt to identify speakers.\n\n" + "\n".join(meta) + "\n", encoding="utf-8")
    return paths


def resolve_ref(cfg: dict, lang: str, allow_download: bool = True) -> tuple[str | None, str | None]:
    """Return (ref_path, warning). ref_path None => use TTS default voice ('no accent reference')."""
    if lang in ("none", "", None):
        return None, None
    t = cfg["tts"]
    refs_dir, lo, hi = cfg["paths"]["refs_dir"], t.get("ref_min_s", 5.0), t.get("ref_max_s", 10.0)
    found = find_refs(refs_dir, lang, lo, hi)
    cv = t.get("common_voice", {})
    if not found and allow_download and cv.get("enabled", False):
        try:
            found = fetch_common_voice(lang, refs_dir, cv.get("repo", "fixie-ai/common_voice_17_0"),
                                       cv.get("split", "test"), 1, lo, hi)
        except Exception as e:  # never block the pipeline on reference download
            log.warning("Common Voice fetch failed for %s: %s", lang, e)
    if found:
        return str(found[0]), None
    return None, (f"No accent reference for '{lang}' (no 5-10 s wav in refs/{lang}/ and Common Voice "
                  "fetch unavailable): using the default TTS voice, so this condition has NO L1 accent.")
