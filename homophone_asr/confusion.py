"""Homophone groups: words whose (stress-stripped) espeak-ng IPA is identical form one group."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

STRESS_RE = re.compile(r"[ˈˌ'‿\-\s]")


class G2P(Protocol):
    name: str

    def phonemize(self, words: list[str]) -> list[str]: ...


class EspeakG2P:
    """phonemizer + espeak-ng backend (language 'de')."""

    name = "espeak"

    def __init__(self, language: str = "de", strip_stress: bool = True):
        self.language, self.strip_stress = language, strip_stress
        self._memo: dict[str, str] = {}

    def phonemize(self, words):
        todo = [w for w in dict.fromkeys(words) if w not in self._memo]
        if todo:
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            out = phonemize(todo, language=self.language, backend="espeak",
                            separator=Separator(phone="", word=" "), strip=True,
                            preserve_punctuation=False, with_stress=not self.strip_stress, njobs=1)
            for w, p in zip(todo, out):
                self._memo[w] = STRESS_RE.sub("", p) if self.strip_stress else p.strip()
        return [self._memo[w] for w in words]


class MockG2P:
    """Crude rule-based German spelling->sound approximation (for offline tests only)."""

    name = "mock"
    RULES = [("dt", "t"), ("ai", "ei"), ("ä", "e"), ("ie", "i"), ("v", "f")]

    def phonemize(self, words):
        out = []
        for w in words:
            s = w.lower()
            for a, b in self.RULES:
                s = s.replace(a, b)
            s = re.sub(r"([aeiouöü])h(?![aeiouöü])", r"\1", s)   # lengthening h
            s = re.sub(r"(.)\1+", r"\1", s)                        # doubled letters
            s = re.sub(r"d$", "t", s)                              # final devoicing
            s = re.sub(r"b$", "p", s)
            s = re.sub(r"g$", "k", s)
            out.append(s)
        return out


@dataclass
class HomophoneGroup:
    ipa: str
    words: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"ipa": self.ipa, "words": self.words}


def build_groups(words: Iterable[str], g2p: G2P) -> list[HomophoneGroup]:
    """Group words by identical IPA. Case variants (Weise/weise) count as ONE spelling.

    Only groups with >= 2 distinct lower-cased spellings are returned (sorted, deterministic).
    """
    uniq: dict[str, str] = {}
    for w in words:
        w = w.strip()
        if w and w.lower() not in uniq:
            uniq[w.lower()] = w
    forms = sorted(uniq.values(), key=lambda s: (s.lower(), s))
    by_ipa: dict[str, list[str]] = {}
    for w, ipa in zip(forms, g2p.phonemize(forms)):
        if ipa:
            by_ipa.setdefault(ipa, []).append(w)
    groups = [HomophoneGroup(ipa, ws) for ipa, ws in by_ipa.items() if len(ws) >= 2]
    return sorted(groups, key=lambda g: g.words[0].lower())


class Lexicon:
    """Case-insensitive lookup word -> homophone group / partners."""

    def __init__(self, groups: list[HomophoneGroup]):
        self.groups = groups
        self._idx = {w.lower(): g for g in groups for w in g.words}

    def group_of(self, word: str) -> HomophoneGroup | None:
        return self._idx.get(word.lower())

    def partners(self, word: str) -> list[str]:
        g = self.group_of(word)
        return [w for w in g.words if w.lower() != word.lower()] if g else []

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._idx

    def partner_map(self) -> dict[str, list[str]]:
        return {w: self.partners(w) for g in self.groups for w in g.words}


def read_sentences(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["target"] not in r["group"].split("|"):
            raise ValueError(f"row {r['id']}: target {r['target']!r} not in group {r['group']!r}")
    return rows


def read_words(lexicon_path: str | Path | None, sentences: list[dict] | None = None) -> list[str]:
    words: list[str] = []
    for r in sentences or []:
        words += r["group"].split("|")
    if lexicon_path and Path(lexicon_path).exists():
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                words.append(line)
    return words


def make_g2p(cfg: dict) -> G2P:
    if cfg["backends"].get("g2p", "espeak") == "mock":
        return MockG2P()
    g = cfg.get("g2p", {})
    return EspeakG2P(g.get("language", "de"), g.get("strip_stress", True))


def load_lexicon(cfg: dict, sentences: list[dict] | None = None) -> Lexicon:
    if sentences is None:
        sentences = read_sentences(cfg["paths"]["sentences"])
    words = read_words(cfg["paths"].get("lexicon"), sentences)
    return Lexicon(build_groups(words, make_g2p(cfg)))


def check_declared_groups(sentences: list[dict], lex: Lexicon) -> list[str]:
    """Return declared groups (CSV) that the G2P does NOT consider homophones."""
    bad = []
    for grp in sorted({r["group"] for r in sentences}):
        ws = grp.split("|")
        g = lex.group_of(ws[0])
        if g is None or any(lex.group_of(w) is not g for w in ws):
            bad.append(grp)
    return bad
