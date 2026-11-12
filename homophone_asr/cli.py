"""Command line interface.

  homophone-asr run --pairs data/sentences_de.csv --accents tr,ru,en --cfg 0,0.3,0.5
  homophone-asr groups [--words mehr,Meer,Rad,Rat]
  homophone-asr fetch-refs --langs tr,ru,ar,en,de
  homophone-asr try "Das Meer ist tief." --accent tr --cfg 0.5
  homophone-asr detect "Ich will das Meer Wasser."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import load_config, mock_overrides


def _cfg_weight(x: str) -> float:
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {x!r}")
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"cfg weight {v} outside [0, 1]")
    return v


def _floats(s: str) -> list[float]:
    vals = [_cfg_weight(x) for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("empty list")
    return vals


def _list(s: str) -> list[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("empty list")
    return vals


def _pos_int(x: str) -> int:
    v = int(x)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _cfg(args) -> dict:
    ov: dict = {}
    if getattr(args, "mock", False):
        ov.update(mock_overrides())
    if getattr(args, "asr_model", None):
        ov.setdefault("asr", {})["model"] = args.asr_model
    if getattr(args, "pairs", None):
        ov.setdefault("paths", {})["sentences"] = args.pairs
    cfg = load_config(args.config, ov)
    if getattr(args, "pairs", None):  # CLI paths are relative to CWD, not the config file
        from pathlib import Path
        cfg["paths"]["sentences"] = str(Path(args.pairs).resolve())
    if getattr(args, "no_cache", False):
        cfg["_no_cache"] = True
    return cfg


def _pipeline(cfg):
    from .pipeline import Pipeline
    p = Pipeline(cfg)
    if cfg.get("_no_cache"):
        p.cache.enabled = False
    return p


def cmd_run(args) -> int:
    from .pipeline import save_results
    cfg = _cfg(args)
    p = _pipeline(cfg)
    rows = p.select(args.pair_ids, args.limit_pairs)
    if not rows:
        print(f"error: no sentences selected (pair ids: {args.pair_ids})", file=sys.stderr)
        return 2
    accents = args.accents if args.accents is not None else cfg["experiment"]["accents"]
    cfgs = args.cfg if args.cfg is not None else cfg["experiment"]["cfg_weights"]
    for a in accents:
        if p.ref_for(a)[1]:
            print("WARNING:", p.ref_for(a)[1], file=sys.stderr)

    def prog(frac, desc):
        print(f"\r[{frac:5.1%}] {desc[:90]:<90}", end="", file=sys.stderr, flush=True)

    read_as = args.read_as or cfg["tts"].get("read_as", "de")
    res = p.run_experiment(accents, cfgs, rows, args.mode, prog, read_as)
    print(file=sys.stderr)
    name = args.name or f"{'mock' if p.is_mock else 'run'}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = save_results(res, cfg["paths"]["results_dir"], name)
    print((d / "report.md").read_text(encoding="utf-8"))
    print(f"saved to {d}")
    return 0


def cmd_groups(args) -> int:
    from .confusion import Lexicon, build_groups, load_lexicon, make_g2p
    cfg = _cfg(args)
    lex = Lexicon(build_groups(args.words, make_g2p(cfg))) if args.words else load_lexicon(cfg)
    for g in lex.groups:
        print(f"/{g.ipa}/\t" + " | ".join(g.words))
    return 0


def cmd_fetch_refs(args) -> int:
    from .refs import fetch_common_voice
    cfg = _cfg(args)
    cv = cfg["tts"]["common_voice"]
    rc = 0
    for lang in args.langs:
        try:
            print(lang, [str(x) for x in fetch_common_voice(lang, cfg["paths"]["refs_dir"], cv["repo"],
                                                            cv["split"], args.n)])
        except Exception as e:
            print(f"{lang}: FAILED ({e}) – the pipeline will fall back to the default voice", file=sys.stderr)
            rc = 1
    return rc


def cmd_try(args) -> int:
    p = _pipeline(_cfg(args))
    r = p.run_sentence(args.text, args.accent, args.cfg, read_as=args.read_as or p.cfg["tts"].get("read_as", "de"))
    if r.get("warning"):
        print("WARNING:", r["warning"], file=sys.stderr)
    print(json.dumps({k: r[k] for k in ("hyp", "read_as", "wer", "target", "recognized", "status", "audio_path")},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_detect(args) -> int:
    cfg = _cfg(args)
    if args.threshold is not None:
        cfg["detector"]["threshold"] = args.threshold
    p = _pipeline(cfg)
    for f in p.detector.detect(args.text):
        mark = "FLAG" if f.flagged else "ok  "
        print(f"{mark} {f.word!r} -> {f.best!r}  margin={f.margin:+.2f}  scores={f.scores}")
    return 0


def cmd_detect_eval(args) -> int:
    from .detector import evaluate_detector
    cfg = _cfg(args)
    if args.threshold is not None:
        cfg["detector"]["threshold"] = args.threshold
    p = _pipeline(cfg)
    e = evaluate_detector(p.detector, p.sentences)
    for r in e["rows"]:
        if not r["correct"]:
            print(f"MISS {r['kind']:8s} {r['text']!r} best={r['best']} margin={r['margin']:+.2f}")
    print(json.dumps({k: v for k, v in e.items() if k != "rows"}, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="homophone-asr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config YAML (default: repo config.yaml)")
    ap.add_argument("--mock", action="store_true", help="use Mock backends (no GPU/downloads)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the sentence set across accents x accent strengths")
    r.add_argument("--pairs", default=None, help="sentence CSV (default from config)")
    r.add_argument("--accents", type=_list, default=None, help="comma list of reference languages, e.g. tr,ru,en")
    r.add_argument("--cfg", type=_floats, default=None, help="comma list of cfg weights, e.g. 0,0.3,0.5")
    r.add_argument("--mode", choices=["sentences", "words", "both"], default="sentences")
    r.add_argument("--limit-pairs", type=_pos_int, default=None)
    r.add_argument("--pair-ids", type=_list, default=None, help="comma list, e.g. p01,p02")
    r.add_argument("--asr-model", default=None, help="override asr.model (e.g. large-v3)")
    r.add_argument("--name", default=None, help="results sub-directory name")
    r.add_argument("--read-as", choices=["de", "accent"], default=None,
                   help="read text with German rules (default) or the accent language's rules (stronger)")
    r.add_argument("--no-cache", action="store_true")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("groups", help="print homophone groups (espeak-ng IPA)")
    g.add_argument("--words", type=_list, default=None, help="comma list; default: dataset + lexicon")
    g.set_defaults(fn=cmd_groups)

    f = sub.add_parser("fetch-refs", help="download reference clips from Common Voice (CC0)")
    f.add_argument("--langs", type=_list, default=["tr", "ru", "ar", "en", "de"])
    f.add_argument("-n", type=int, default=1)
    f.set_defaults(fn=cmd_fetch_refs)

    t = sub.add_parser("try", help="synthesize + transcribe one sentence")
    t.add_argument("text")
    t.add_argument("--accent", default="tr")
    t.add_argument("--cfg", type=_cfg_weight, default=0.5)
    t.add_argument("--read-as", choices=["de", "accent"], default=None)
    t.set_defaults(fn=cmd_try)

    d = sub.add_parser("detect", help="flag context-implausible homophones in text")
    d.add_argument("text")
    d.add_argument("--threshold", type=float, default=None)
    d.set_defaults(fn=cmd_detect)
    e = sub.add_parser("detect-eval", help="evaluate detector: originals vs. target->partner swaps")
    e.add_argument("--threshold", type=float, default=None)
    e.set_defaults(fn=cmd_detect_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
