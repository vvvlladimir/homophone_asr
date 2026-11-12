"""Gradio demo: python app.py [--mock] [--config config.yaml] [--port 7860] [--share]

Tabs: "Try it" (one sentence), "Experiment" (sentence set x accents x strengths), "Detector".
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from homophone_asr import analyze  # noqa: E402
from homophone_asr.config import load_config, mock_overrides  # noqa: E402
from homophone_asr.pipeline import Pipeline, save_results  # noqa: E402

log = logging.getLogger("app")

PRESET = {"text": "Das Meer ist heute sehr ruhig.", "accent": "tr", "cfg": 0.5}
EXAMPLES = [["Die tiefe Saite klingt verstimmt.", "tr", 0.5], ["Mein linkes Lid zuckt ständig.", "de", 0.5],
            ["Im Ozean schwimmt ein großer Wal.", "tr", 0.5], ["Das Meer ist heute sehr ruhig.", "ru", 0.5],
            ["Danke für deinen guten Rat.", "ru", 0.5], ["Ich möchte bitte mehr Wasser.", "de", 0.5]]
DETECT_EXAMPLES = ["Ich will Meer Wasser trinken.", "Ich glaube, das er morgen kommt.",
                   "Er fährt jeden Tag mit dem Rat zur Arbeit.", "Wir wohnen mitten in der statt."]
UNVALIDATED = ("**Synthetic accents.** The accent comes from cross-lingual voice prompting "
               "(German text + a reference clip in another language, Chatterbox Multilingual). "
               "It is **not validated** against real L2 speech – treat results as a stress test, "
               "not as a measurement of how real learners are recognized.")
DIFF_COLORS = {"homophone": "#d95926", "sub": "#e8b04b", "del": "#9b9b9b", "ins": "#6da7ec",
               "target ok": "#5fb36b"}
CMAP_BLUE = LinearSegmentedColormap.from_list("blue", ["#eef5fd", "#86b6ef", "#2a78d6", "#0d366b"])
CMAP_ORANGE = LinearSegmentedColormap.from_list("orange", ["#fdf0ea", "#f2a07f", "#eb6834", "#8a3210"])
STATUS_TEXT = {
    analyze.CORRECT: "✅ Target homophone recognized correctly",
    analyze.HOMOPHONE: "⚠️ **Homophone confusion**",
    analyze.OTHER: "❌ Target replaced by a non-homophone word",
    analyze.DELETED: "❌ Target word missing in the transcript",
}


# --------------------------------------------------------------------------- helpers
def accent_choices(cfg) -> list[tuple[str, str]]:
    acc = cfg["tts"].get("accents", {})
    return [(f"{name} [{code}]", code) for code, name in acc.items()] + \
        [("Default voice, no reference [none]", "none")]


def heatmaps(summary: pd.DataFrame):
    if summary is None or summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 * summary["accent"].nunique() + 2.2))
    for ax, col, title, cmap in ((axes[0], "homophone_confusion_rate", "Homophone-confusion rate", CMAP_ORANGE),
                                 (axes[1], "wer", "WER", CMAP_BLUE)):
        piv = summary.pivot_table(index="accent", columns="cfg_weight", values=col)
        vmax = max(0.05, float(piv.max().max()))
        im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel("cfg_weight (accent strength)")
        ax.set_title(title, fontsize=11, loc="left")
        for (i, j), v in pd.DataFrame(piv.values).stack().items():
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.6 * vmax else "#222222")
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    return fig


def items_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["accent", "cfg_weight", "sentence", "hyp", "target", "recognized", "status", "wer", "ref_used"]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[[c for c in cols if c in df.columns]].round({"wer": 3})


def directions_md(d: dict) -> str:
    if not d:
        return "_No homophone confusions in this run._"
    return "**Confusion directions (target → recognized):** " + ", ".join(f"`{k}` ×{v}" for k, v in d.items())


# --------------------------------------------------------------------------- app
def build_app(pipe: Pipeline) -> gr.Blocks:
    cfg = pipe.cfg
    choices = accent_choices(cfg)
    gloss = {r["target"].lower(): r["gloss"] for r in pipe.sentences}
    mock_note = ("\n\n> 🧪 **MOCK MODE** – TTS/ASR/LM are simulations; numbers are meaningless."
                 if pipe.is_mock else "")

    # ---- Try it
    def try_it(text, accent, cfg_weight, progress=gr.Progress()):
        text = (text or "").strip()
        if not text:
            gr.Warning("Please type a German sentence.")
            return None, "", [], "", [], "_Type a German sentence first._", None
        progress(0.1, desc="Synthesizing (≈25 s per new sentence on CPU; cached results are instant)…")
        r = pipe.run_sentence(text, accent, float(cfg_weight))
        notes = []
        if r.get("warning"):
            notes.append(f"⚠️ **No accent reference:** {r['warning']}")
        elif r.get("ref_path"):
            notes.append(f"Reference clip: `{Path(r['ref_path']).parent.name}/{Path(r['ref_path']).name}` "
                         f"(Common Voice, CC0)")
        notes.append("cached" if r["tts_cached"] and r["asr_cached"] else "freshly generated")
        if r["status"] is None:
            verdict = "No known homophone in this sentence – showing plain WER only."
        else:
            verdict = STATUS_TEXT.get(r["status"], r["status"])
            if r["status"] == analyze.HOMOPHONE:
                g = lambda w: f" ({gloss[w.lower()]})" if w and w.lower() in gloss else ""  # noqa: E731
                verdict += (f": said **{r['target']}**{g(r['target'])}, "
                            f"ASR wrote **{r['recognized']}**{g(r['recognized'])}")
        verdict += f"  \nWER = {r['wer']:.0%} ({r['errors']} errors / {r['n_ref']} words)"
        spans = analyze.diff_spans(r["ops"], r["target_idx"], r["status"], r["recognized"])
        det = pipe.detector
        hyp = r["hyp"]
        det_spans = det.highlight(hyp, det.detect(hyp)) if hyp else []
        ref_audio = r.get("ref_path")
        return (r["audio_path"], hyp, spans, verdict, det_spans, " · ".join(notes), ref_audio)

    # ---- Experiment
    pair_labels = {}
    for r in pipe.sentences:
        pair_labels.setdefault(r["group"].replace("|", " / "), r["pair_id"])

    def run_exp(accents, cfgs, pairs, mode, progress=gr.Progress()):
        if not accents or not cfgs or not pairs:
            gr.Warning("Select at least one accent, one strength and one pair.")
            return ("⚠️ _Select at least one accent, one strength and one pair – previous results kept._",
                    *[gr.skip()] * 5)
        rows = pipe.select([pair_labels[p] for p in pairs])
        res = pipe.run_experiment(list(accents), sorted(float(c) for c in cfgs), rows, mode,
                                  lambda f, d: progress(f, desc=d))
        save_results(res, cfg["paths"]["results_dir"], "last_ui_run" + ("_mock" if pipe.is_mock else ""))
        return exp_outputs(res["summary"], res["word_summary"], res["sentences"] + res["words"],
                           res["directions"], "Fresh run (saved to results/last_ui_run).", res["meta"])

    def exp_outputs(summary, wsum, rows, directions, title, meta):
        no_ref = [a for a, ref in (meta.get("refs") or {}).items() if ref is None and a != "none"]
        info = f"**{title}**" + (" – MOCK numbers" if meta.get("mock") else "")
        info += f"  \nASR: `{meta.get('asr', {}).get('model', meta.get('asr', {}).get('backend'))}` · " \
                f"accents: {meta.get('accents')} · cfg: {meta.get('cfg_weights')} · {meta.get('seconds')} s"
        if no_ref:
            info += f"  \n⚠️ No accent reference for {no_ref}: those rows used the default voice."
        return (info, summary, heatmaps(summary), wsum, items_table(rows), directions_md(directions))

    def load_saved():
        for name in (["last_ui_run_mock"] if pipe.is_mock else ["last_ui_run", "smoke"]):
            p = Path(cfg["paths"]["results_dir"]) / name / "rows.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                ok_s = [r for r in d["sentences"] if "error" not in r]
                ok_w = [r for r in d["words"] if "error" not in r]
                return exp_outputs(analyze.aggregate(ok_s), analyze.aggregate_words(ok_w),
                                   d["sentences"] + d["words"], analyze.confusion_directions(ok_s + ok_w),
                                   f"Saved run: results/{name}", d["meta"])
        if pipe.is_mock:  # instant preset in mock mode
            res = pipe.run_experiment(["de", "tr", "ar"], [0.0, 0.5], pipe.select(limit_pairs=3), "sentences")
            return exp_outputs(res["summary"], res["word_summary"], res["sentences"], res["directions"],
                               "Preset run", res["meta"])
        return ("_No saved run yet – press **Run experiment**._", None, None, None, None, "")

    # ---- Detector
    def detect(text, threshold):
        text = text or ""
        fs = pipe.detector.detect(text, float(threshold))   # per-call threshold: no shared state
        table = pd.DataFrame([{"word": f.word, "flagged": f.flagged, "suggestion": f.best,
                               "margin (nats)": round(f.margin, 2),
                               "P(original)": round(f.probs[f.word], 3),
                               "P(suggestion)": round(f.probs[f.best], 3)} for f in fs])
        if not fs:
            msg = "_No word from the homophone lexicon in this text._"
        else:
            n = sum(f.flagged for f in fs)
            msg = f"{len(fs)} homophone(s) checked, **{n} suspicious** (threshold {threshold:.1f} nats)."
        return pipe.detector.highlight(text, fs), table, msg

    with gr.Blocks(title="Homophone ASR stress test") as demo:
        gr.Markdown("# German homophones × synthetic accents × ASR\n"
                    "TTS (Chatterbox Multilingual) → ASR (faster-whisper) → error analysis → "
                    "masked-LM homophone detector (gbert-base). " + mock_note)
        gr.Markdown(UNVALIDATED)

        with gr.Tab("Try it"):
            with gr.Row():
                with gr.Column(scale=3):
                    t_text = gr.Textbox(PRESET["text"], label="German sentence", lines=1)
                    with gr.Row():
                        t_acc = gr.Dropdown(choices, value=PRESET["accent"], label="Accent (reference-clip language)")
                        t_cfg = gr.Slider(0.0, 0.5, value=PRESET["cfg"], step=0.05,
                                          label="Accent strength (cfg_weight; 0 = suppressed)")
                    t_btn = gr.Button("▶ Synthesize & recognize", variant="primary")
                    gr.Examples(EXAMPLES, inputs=[t_text, t_acc, t_cfg],
                                label="Examples (cached from the smoke run → instant)")
                with gr.Column(scale=2):
                    t_audio = gr.Audio(label="Synthesized speech", type="filepath", interactive=False)
                    t_ref = gr.Audio(label="Reference clip (accent source)", type="filepath", interactive=False)
                    t_note = gr.Markdown()
            t_hyp = gr.Textbox(label="ASR transcript", interactive=False)
            t_verdict = gr.Markdown()
            t_diff = gr.HighlightedText(label="Aligned diff (reference→ASR)", color_map=DIFF_COLORS,
                                        show_legend=True, combine_adjacent=False)
            t_det = gr.HighlightedText(label="Detector on the ASR transcript",
                                       color_map={"homophone ok": "#5fb36b"}, show_legend=False)
            t_btn.click(try_it, [t_text, t_acc, t_cfg], [t_audio, t_hyp, t_diff, t_verdict, t_det, t_note, t_ref])

        with gr.Tab("Experiment"):
            with gr.Row():
                e_acc = gr.CheckboxGroup(choices, value=["de", "tr", "ru"], label="Accents")
                e_cfg = gr.CheckboxGroup([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], value=[0.0, 0.5], label="Accent strengths")
            with gr.Row():
                e_pairs = gr.Dropdown(list(pair_labels), value=list(pair_labels)[:2], multiselect=True,
                                      label="Homophone pairs (4 sentences each)")
                e_mode = gr.Radio(["sentences", "words", "both"], value="sentences",
                                  label="Mode (words: isolated, all group members accepted)")
            gr.Markdown("On CPU every *new* clip costs ≈25 s (e.g. 2 pairs × 3 accents × 2 strengths = 48 clips "
                        "≈ 20 min). Cached clips are instant.")
            e_btn = gr.Button("Run experiment", variant="primary")
            e_info = gr.Markdown()
            e_plot = gr.Plot(label="Heatmaps (accent × strength)", show_label=False)
            e_sum = gr.Dataframe(label="Sentence summary (context test)", interactive=False)
            e_dir = gr.Markdown()
            e_wsum = gr.Dataframe(label="Isolated-word summary – ⚠ unreliable: Chatterbox/Whisper fail on 1-word clips "
                                          "(see results/smoke.md)", interactive=False)
            e_items = gr.Dataframe(label="All items", interactive=False, wrap=True)
            e_btn.click(run_exp, [e_acc, e_cfg, e_pairs, e_mode], [e_info, e_sum, e_plot, e_wsum, e_items, e_dir])

        with gr.Tab("Detector"):
            d_text = gr.Textbox(DETECT_EXAMPLES[0], label="German text (e.g. an ASR transcript)", lines=3)
            d_thr = gr.Slider(0.0, 8.0, value=float(cfg["detector"].get("threshold", 2.0)), step=0.25,
                              label="Threshold: flag if log P(partner) − log P(original) ≥ this")
            d_btn = gr.Button("Check homophones", variant="primary")
            gr.Examples([[t] for t in DETECT_EXAMPLES], inputs=[d_text])
            d_msg = gr.Markdown()
            d_out = gr.HighlightedText(label="Homophones checked (suspicious ones show the suggested word)", color_map={"homophone ok": "#5fb36b"})
            d_tab = gr.Dataframe(label="Scores", interactive=False)
            d_btn.click(detect, [d_text, d_thr], [d_out, d_tab, d_msg])

        demo.load(load_saved, None, [e_info, e_sum, e_plot, e_wsum, e_items, e_dir])
        demo.load(detect, [d_text, d_thr], [d_out, d_tab, d_msg])
        demo.load(lambda: try_it(PRESET["text"], PRESET["accent"], PRESET["cfg"]) if _ready.is_set() else
                  [gr.skip()] * 7, None, [t_audio, t_hyp, t_diff, t_verdict, t_det, t_note, t_ref])
    return demo


_ready = threading.Event()


def warmup(pipe: Pipeline):
    """Load models and pre-compute the preset so the first click is instant."""
    try:
        for m in (pipe.tts, pipe.asr):
            if hasattr(m, "load"):
                m.load()
        pipe.detector.detect(DETECT_EXAMPLES[0])
        pipe.run_sentence(PRESET["text"], PRESET["accent"], PRESET["cfg"])
        log.warning("warm-up done: preset cached")
    except Exception:
        log.exception("warm-up failed (the demo still works, first click will be slower)")
    finally:
        _ready.set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Mock backends (no models, instant)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(a.config, mock_overrides() if a.mock else None)
    pipe = Pipeline(cfg)
    if a.no_warmup:
        _ready.set()
    else:
        threading.Thread(target=warmup, args=(pipe,), daemon=True).start()
    build_app(pipe).queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, share=a.share,
        allowed_paths=[cfg["paths"]["cache_dir"], cfg["paths"]["refs_dir"]])


if __name__ == "__main__":
    main()
