"""Context-based homophone detector (masked LM).

Interface: `MaskedLM.candidate_logprobs(text, span, candidates) -> {candidate: log p}` where `span`
is the (start, end) character span of the word to replace. The detector compares the original word
with its homophone partners in context and flags it if a partner is clearly more likely.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

WORD_RE = re.compile(r"\w+", re.UNICODE)


class MaskedLM(Protocol):
    name: str

    def candidate_logprobs(self, text: str, span: tuple[int, int],
                           candidates: list[str]) -> dict[str, float]: ...


class MockLM:
    """Tiny co-occurrence 'LM' built from a reference corpus (e.g. the sentence CSV).

    score(c) ~ how many context words (window) co-occur with c in the corpus. Deterministic, offline.
    """

    name = "mock"

    def __init__(self, corpus: list[str], window: int = 4):
        self.window = window
        self.sents = [set(w.lower() for w in WORD_RE.findall(s)) for s in corpus]

    def candidate_logprobs(self, text, span, candidates):
        words = [(m.group(0).lower(), m.start()) for m in WORD_RE.finditer(text)]
        idx = next((i for i, (_, s) in enumerate(words) if s == span[0]), None)
        if idx is None:
            ctx = set()
        else:
            lo, hi = max(0, idx - self.window), idx + self.window + 1
            ctx = {w for j, (w, _) in enumerate(words[lo:hi], lo) if j != idx}
        raw = {}
        for c in candidates:
            cl = c.lower()
            raw[c] = 1.0 + sum(len(ctx & (s - {cl})) for s in self.sents if cl in s)
        z = sum(raw.values())
        return {c: math.log(v / z) for c, v in raw.items()}


class HFMaskedLM:
    """Hugging Face masked LM (default deepset/gbert-base).

    Candidate score = word-level pseudo-log-likelihood with the 'l2r' fix (Kauf & Ivanova 2023):
    for subword j of the candidate, mask j and all later subwords of the same word, sum log p.
    All masked variants of all candidates go through the model in one batch.
    """

    name = "hf"

    def __init__(self, model: str = "deepset/gbert-base", device: str = "cpu", max_ctx: int = 200):
        self.model_name, self.device, self.max_ctx = model, device, max_ctx
        self._tok = self._model = None

    def load(self):
        if self._model is None:
            from transformers import (AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM,
                                      BertTokenizerFast)
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            try:
                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            except (ValueError, KeyError, OSError):
                # deepset/gbert-base ships a config without `model_type`: load as plain BERT,
                # cased and WITHOUT accent stripping (keeps umlauts/ß).
                self._tok = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=False,
                                                              strip_accents=False)
                model = BertForMaskedLM.from_pretrained(self.model_name)
            self._model = model.to(self.device).eval()
        return self._tok, self._model

    def candidate_logprobs(self, text, span, candidates):
        import torch
        tok, model = self.load()
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        pre, suf = enc(text[:span[0]])[-self.max_ctx:], enc(text[span[1]:])[:self.max_ctx]
        inputs, targets, owner = [], [], []
        for ci, c in enumerate(candidates):
            ids = enc(c)
            base = [tok.cls_token_id] + pre + ids + suf + [tok.sep_token_id]
            p0 = 1 + len(pre)
            for j in range(len(ids)):
                x = list(base)
                for k in range(j, len(ids)):
                    x[p0 + k] = tok.mask_token_id
                inputs.append(x)
                targets.append((p0 + j, ids[j]))
                owner.append(ci)
        L = max(len(x) for x in inputs)
        pad = tok.pad_token_id or 0
        ids_t = torch.tensor([x + [pad] * (L - len(x)) for x in inputs], device=self.device)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in inputs], device=self.device)
        with torch.inference_mode():
            logits = model(input_ids=ids_t, attention_mask=att).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        out = [0.0] * len(candidates)
        for row, ((pos, tid), ci) in enumerate(zip(targets, owner)):
            out[ci] += float(logp[row, pos, tid])
        return dict(zip(candidates, out))


def make_lm(cfg: dict, sentences: list[dict] | None = None) -> MaskedLM:
    if cfg["backends"]["lm"] == "mock":
        return MockLM([r["sentence"] for r in sentences or []])
    d = cfg["detector"]
    return HFMaskedLM(d.get("model", "deepset/gbert-base"), d.get("device", "cpu"))


@dataclass
class Finding:
    word: str
    start: int
    end: int
    scores: dict[str, float]          # candidate -> log p (original under its written form)
    probs: dict[str, float]           # softmax over candidates (for display)
    best: str                         # most likely partner
    margin: float                     # log p(best partner) - log p(original)
    flagged: bool


def _match_case(word: str, like: str, sentence_start: bool) -> str:
    if like.isupper() and len(like) > 1:
        return word.replace("ß", "ẞ").upper()   # keep ß distinct from ss (Maß vs Mass)
    if sentence_start:
        return word[:1].upper() + word[1:]
    return word


def _is_sentence_start(text: str, start: int) -> bool:
    before = text[:start].rstrip().rstrip("„“\"«»‚'(").rstrip()   # skip OPENING quotes/brackets
    return not before or before[-1] in ".!?:"


class HomophoneDetector:
    def __init__(self, lm: MaskedLM, lexicon, threshold: float = 2.0):
        self.lm, self.lexicon, self.threshold = lm, lexicon, float(threshold)

    def check(self, text: str, start: int, end: int, threshold: float | None = None) -> Finding | None:
        word = text[start:end]
        partners = self.lexicon.partners(word)
        if not partners:
            return None
        s0 = _is_sentence_start(text, start)
        g = self.lexicon.group_of(word)
        canon = next(w for w in g.words if w.lower() == word.lower())
        orig_forms = list(dict.fromkeys([word, _match_case(canon, word, s0)]))
        part_forms = [_match_case(p, word, s0) for p in partners]
        cands = list(dict.fromkeys(orig_forms + part_forms))
        lp = self.lm.candidate_logprobs(text, (start, end), cands)
        # original = best of {as written, canonical}; partners use canonical spelling only.
        # Deliberately conservative: casing errors in ASR output alone never trigger a flag.
        orig = max(lp[f] for f in orig_forms)
        part = {p: lp[f] for p, f in zip(partners, part_forms)}
        best = max(part, key=lambda k: (part[k], k))
        scores = {word: orig, **part}
        m = max(scores.values())
        z = sum(math.exp(v - m) for v in scores.values())
        probs = {k: math.exp(v - m) / z for k, v in scores.items()}
        margin = part[best] - orig
        thr = self.threshold if threshold is None else float(threshold)
        return Finding(word, start, end, scores, probs, best, margin, margin >= thr)

    def detect(self, text: str, threshold: float | None = None) -> list[Finding]:
        out = []
        for mt in WORD_RE.finditer(text or ""):
            f = self.check(text, mt.start(), mt.end(), threshold)
            if f:
                out.append(f)
        return out

    @staticmethod
    def highlight(text: str, findings: list[Finding]) -> list[tuple[str, str | None]]:
        """(segment, label) pairs for gr.HighlightedText."""
        out, pos = [], 0
        for f in findings:
            if f.start > pos:
                out.append((text[pos:f.start], None))
            if f.flagged:
                label = f"suspicious → {f.best} (+{f.margin:.1f})"
            else:
                label = "homophone ok"
            out.append((text[f.start:f.end], label))
            pos = f.end
        if pos < len(text):
            out.append((text[pos:], None))
        return out


def _find_word(sentence: str, target: str) -> tuple[int, int] | None:
    for mt in WORD_RE.finditer(sentence):
        if mt.group(0).lower() == target.lower():
            return mt.start(), mt.end()
    return None


def swap_target(sentence: str, target: str, replacement: str) -> str | None:
    """Replace the first whole-word occurrence of target (case-insensitive), keeping sentence-start case."""
    sp = _find_word(sentence, target)
    if sp is None:
        return None
    rep = _match_case(replacement, sentence[sp[0]:sp[1]], _is_sentence_start(sentence, sp[0]))
    return sentence[:sp[0]] + rep + sentence[sp[1]:]


def evaluate_detector(det: HomophoneDetector, sentences: list[dict]) -> dict:
    """Original sentences should NOT be flagged at the target (each counted once); every
    target->partner swap SHOULD be flagged with the original target suggested.
    The checked word is located by its character span, not by first match in detect()."""
    rows, skipped = [], 0
    for r in sentences:
        sp = _find_word(r["sentence"], r["target"])
        if sp is None or not det.lexicon.partners(r["target"]):
            skipped += 1
            continue
        items = [("original", r["sentence"], False)]
        for partner in det.lexicon.partners(r["target"]):
            items.append(("swapped", swap_target(r["sentence"], r["target"], partner), True))
        for kind, text, want in items:
            # the swap keeps everything before the target, so the span start is unchanged
            end = WORD_RE.match(text, sp[0]).end()
            f = det.check(text, sp[0], end)
            if f is None:
                skipped += 1
                continue
            ok_sugg = (not want) or f.best.lower() == r["target"].lower()
            rows.append({"id": r["id"], "kind": kind, "text": text, "word": f.word, "best": f.best,
                         "margin": round(f.margin, 3), "flagged": f.flagged,
                         "correct": f.flagged == want and ok_sugg})
    sw = [x for x in rows if x["kind"] == "swapped"]
    og = [x for x in rows if x["kind"] == "original"]
    return {"detection_rate": sum(x["correct"] for x in sw) / max(len(sw), 1),
            "false_alarm_rate": sum(x["flagged"] for x in og) / max(len(og), 1),
            "n_swapped": len(sw), "n_original": len(og), "skipped": skipped,
            "threshold": det.threshold, "rows": rows}
