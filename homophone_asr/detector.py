"""Context-based homophone detector (masked LM).

Interface: `MaskedLM.candidate_logprobs(text, span, candidates) -> {candidate: log p}` where `span`
is the (start, end) character span of the word to replace. The detector compares the original word
with its homophone partners in context and flags it if a partner is clearly more likely.
"""
from __future__ import annotations

import math
import re
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
