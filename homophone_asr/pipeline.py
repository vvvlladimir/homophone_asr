"""Pipeline wiring: config -> backends -> (TTS -> ASR -> analysis), with content-hash caching."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from . import analyze
from .asr import ASRBackend, make_asr
from .cache import Cache, content_hash, file_hash
from .confusion import Lexicon, load_lexicon, read_sentences
from .refs import resolve_ref
from .tts import TTSBackend, make_tts

log = logging.getLogger(__name__)
NO_REF_LABEL = "none"   # accent key meaning: TTS default voice, no reference clip


class Pipeline:
    def __init__(self, cfg: dict, tts: TTSBackend | None = None, asr: ASRBackend | None = None,
                 lexicon: Lexicon | None = None, sentences: list[dict] | None = None,
                 allow_download: bool = True):
        self.cfg = cfg
        self.sentences = sentences if sentences is not None else read_sentences(cfg["paths"]["sentences"])
        self.lexicon = lexicon or load_lexicon(cfg, self.sentences)
        self.tts = tts or make_tts(cfg)
        self.asr = asr or make_asr(cfg, self.lexicon.partner_map())
        self.cache = Cache(cfg["paths"]["cache_dir"])
        self.allow_download = allow_download
        self._refs: dict[str, tuple[str | None, str | None]] = {}
        self._detector = None

    @property
    def is_mock(self) -> bool:
        return self.tts.name == "mock" or self.asr.name == "mock"

    # ---------- building blocks ----------
    def ref_for(self, accent: str) -> tuple[str | None, str | None]:
        if accent not in self._refs:
            self._refs[accent] = resolve_ref(self.cfg, accent, self.allow_download)
        return self._refs[accent]

    def synthesize(self, text: str, accent: str, cfg_weight: float) -> dict:
        ref, warning = self.ref_for(accent)
        key = content_hash({"text": text, "ref": file_hash(ref) if ref else None,
                            "cfg_weight": round(float(cfg_weight), 3), "tts": self.tts.cache_params()})
        path = self.cache.path("audio", key, "wav")
        cached = path.exists() and self.cache.enabled
        if not cached:
            tmp = path.with_name(path.stem + ".part.wav")
            self.tts.synthesize(text, tmp, accent, ref, float(cfg_weight))
            tmp.replace(path)
        return {"audio_path": str(path), "ref_path": ref, "ref_used": ref is not None,
                "warning": warning, "tts_cached": cached}

    def transcribe(self, audio_path: str) -> dict:
        key = content_hash({"audio": file_hash(audio_path), "asr": self.asr.cache_params()})
        hit = self.cache.get_json("asr", key)
        if hit is not None:
            return {"hyp": hit["hyp"], "asr_cached": True}
        hyp = self.asr.transcribe(audio_path)
        self.cache.put_json("asr", key, {"hyp": hyp, "audio": Path(audio_path).name})
        return {"hyp": hyp, "asr_cached": False}

    def _group(self, target: str, declared: str | None = None) -> list[str]:
        if declared:
            return declared.split("|")
        g = self.lexicon.group_of(target)
        return g.words if g else [target]

    # ---------- single items ----------
    def run_sentence(self, sentence: str, accent: str, cfg_weight: float, target: str | None = None,
                     group: str | None = None, meta: dict | None = None) -> dict:
        syn = self.synthesize(sentence, accent, cfg_weight)
        asr = self.transcribe(syn["audio_path"])
        targets = [target] if target else [w for w in analyze.tokens(sentence) if w in self.lexicon]
        row = {**(meta or {}), "mode": "sentence", "accent": accent, "cfg_weight": float(cfg_weight),
               "sentence": sentence, **syn, **asr}
        if targets:
            a = analyze.analyze_sentence(sentence, asr["hyp"], targets[0], self._group(targets[0], group))
        else:  # free text with no known homophone: plain WER
            ops, counts = analyze.align(sentence, asr["hyp"])
            a = {**counts, "ref_norm": analyze.normalize(sentence), "hyp_norm": analyze.normalize(asr["hyp"]),
                 "target": None, "target_idx": None, "recognized": None, "status": None,
                 "ops": [o.__dict__ for o in ops]}
        row.update(a)
        row["group"] = "|".join(self._group(targets[0], group)) if targets else None
        return row

    def run_word(self, word: str, accent: str, cfg_weight: float, group: str | None = None,
                 meta: dict | None = None) -> dict:
        syn = self.synthesize(word, accent, cfg_weight)
        asr = self.transcribe(syn["audio_path"])
        grp = self._group(word, group)
        return {**(meta or {}), "mode": "word", "accent": accent, "cfg_weight": float(cfg_weight),
                "sentence": word, "group": "|".join(grp), **syn, **asr,
                **analyze.analyze_word(word, asr["hyp"], grp)}

    # ---------- experiments ----------
    def select(self, pair_ids: list[str] | None = None, limit_pairs: int | None = None) -> list[dict]:
        pids = list(dict.fromkeys(r["pair_id"] for r in self.sentences))
        if pair_ids:
            pids = [p for p in pids if p in set(pair_ids)]
        if limit_pairs:
            pids = pids[:limit_pairs]
        return [r for r in self.sentences if r["pair_id"] in set(pids)]

    def run_experiment(self, accents: list[str], cfg_weights: list[float], rows: list[dict] | None = None,
                       mode: str = "sentences", progress: Callable[[float, str], None] | None = None) -> dict:
        rows = rows if rows is not None else self.sentences
        jobs = []
        if mode in ("sentences", "both"):
            jobs += [("s", r) for r in rows]
        if mode in ("words", "both"):
            seen = {}
            for r in rows:
                for w in r["group"].split("|"):
                    seen.setdefault(w.lower(), (w, r))
            jobs += [("w", {"target": w, "group": r["group"], "pair_id": r["pair_id"]}) for w, r in seen.values()]
        total = len(jobs) * len(accents) * len(cfg_weights)
        out_s, out_w, n, t0 = [], [], 0, time.time()
        for acc in accents:
            for cw in cfg_weights:
                for kind, r in jobs:
                    n += 1
                    if progress:
                        progress(n / max(total, 1), f"{acc} cfg={cw} [{n}/{total}] {r.get('sentence', r['target'])}")
                    meta = {"pair_id": r["pair_id"], "id": r.get("id")}
                    try:
                        if kind == "s":
                            out_s.append(self.run_sentence(r["sentence"], acc, cw, r["target"], r["group"], meta))
                        else:
                            out_w.append(self.run_word(r["target"], acc, cw, r["group"], meta))
                    except Exception as e:  # record failures instead of aborting a long run
                        log.exception("item failed")
                        (out_s if kind == "s" else out_w).append({**meta, "accent": acc, "cfg_weight": cw,
                                                                  "error": repr(e)})
        ok_s = [r for r in out_s if "error" not in r]
        ok_w = [r for r in out_w if "error" not in r]
        return {"sentences": out_s, "words": out_w,
                "summary": analyze.aggregate(ok_s), "word_summary": analyze.aggregate_words(ok_w),
                "directions": analyze.confusion_directions(ok_s + ok_w),
                "meta": self.run_meta(accents, cfg_weights, mode, time.time() - t0)}

    def run_meta(self, accents, cfg_weights, mode, seconds) -> dict:
        return {"accents": accents, "cfg_weights": cfg_weights, "mode": mode, "seconds": round(seconds, 1),
                "tts": self.tts.cache_params(), "asr": self.asr.cache_params(), "mock": self.is_mock,
                "refs": {a: (Path(self.ref_for(a)[0]).name if self.ref_for(a)[0] else None) for a in accents}}

    # ---------- detector (phase 3) ----------
    @property
    def detector(self):
        if self._detector is None:
            from .detector import HomophoneDetector, make_lm
            self._detector = HomophoneDetector(make_lm(self.cfg, self.sentences), self.lexicon,
                                               self.cfg["detector"].get("threshold", 2.0))
        return self._detector


def save_results(res: dict, out_dir: str | Path, name: str) -> Path:
    """Write rows (JSON), summaries (CSV) and a short markdown report to out_dir/name/."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.json").write_text(json.dumps({"meta": res["meta"], "sentences": res["sentences"],
                                             "words": res["words"]}, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    res["summary"].to_csv(d / "summary.csv", index=False)
    res["word_summary"].to_csv(d / "word_summary.csv", index=False)
    (d / "report.md").write_text(report_md(res, name), encoding="utf-8")
    return d


def report_md(res: dict, title: str) -> str:
    m = res["meta"]
    lines = [f"# {title}", ""]
    if m.get("mock"):
        lines += ["> **MOCK BACKENDS – simulated numbers, not a measurement.**", ""]
    lines += [f"- TTS: `{m['tts']}`", f"- ASR: `{m['asr']}`", f"- reference clips: `{m['refs']}`",
              f"- runtime: {m['seconds']} s", "",
              "Synthetic accents are produced by cross-lingual voice prompting and are NOT validated "
              "against real L2 speech.", ""]
    if not res["summary"].empty:
        lines += ["## Sentences (context test)", "", res["summary"].to_markdown(index=False), ""]
    if not res["word_summary"].empty:
        lines += ["## Isolated words (all group members accepted)", "",
                  res["word_summary"].to_markdown(index=False), ""]
    if res["directions"]:
        lines += ["## Confusion directions (target → recognized)", ""]
        lines += [f"- {k}: {v}" for k, v in res["directions"].items()] + [""]
    errs = [r for r in res["sentences"] + res["words"] if "error" in r]
    if errs:
        lines += ["## Failed items", ""] + [f"- {r['accent']} {r['cfg_weight']} id={r.get('id')}: {r['error']}"
                                           for r in errs] + [""]
    return "\n".join(lines)
