"""Error analysis: normalization, jiwer word alignment, homophone-confusion classification, aggregation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import jiwer

_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)

# target status values
CORRECT = "correct"
HOMOPHONE = "homophone_confusion"   # target recognized as another member of its group
OTHER = "other_substitution"        # target recognized as a non-member word
DELETED = "deleted"
MISSING = "target_not_in_reference"
# isolated-word status values
EXACT, WITHIN_GROUP = "exact", "within_group"


def normalize(text: str) -> str:
    """NFC, lowercase (NOT casefold: keeps ß), punctuation -> space, collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


@dataclass
class Op:
    op: str            # equal | substitute | delete | insert
    ref_idx: int | None
    hyp_idx: int | None
    ref: str | None
    hyp: str | None


def align(ref: str, hyp: str) -> tuple[list[Op], dict]:
    """Word-level alignment of normalized texts. Returns (ops, counts)."""
    r, h = tokens(ref), tokens(hyp)
    if not r:
        ops = [Op("insert", None, j, None, w) for j, w in enumerate(h)]
    else:
        out = jiwer.process_words(" ".join(r), " ".join(h) if h else "")
        ops = []
        for c in out.alignments[0]:
            if c.type in ("equal", "substitute"):
                for k in range(c.ref_end_idx - c.ref_start_idx):
                    i, j = c.ref_start_idx + k, c.hyp_start_idx + k
                    ops.append(Op(c.type, i, j, r[i], h[j]))
            elif c.type == "delete":
                ops += [Op("delete", i, None, r[i], None) for i in range(c.ref_start_idx, c.ref_end_idx)]
            else:
                ops += [Op("insert", None, j, None, h[j]) for j in range(c.hyp_start_idx, c.hyp_end_idx)]
    cnt = {k: sum(o.op == k for o in ops) for k in ("substitute", "delete", "insert")}
    errors = cnt["substitute"] + cnt["delete"] + cnt["insert"]
    # WER for empty reference: 0 if hyp empty else 1.0 (convention; jiwer refuses empty refs)
    wer = errors / len(r) if r else (1.0 if h else 0.0)
    return ops, {"n_ref": len(r), "n_hyp": len(h), "errors": errors, "wer": wer,
                 "subs": cnt["substitute"], "dels": cnt["delete"], "ins": cnt["insert"]}


def find_target(ref: str, target: str, occurrence: int = 0) -> int | None:
    """Index of the `occurrence`-th normalized token equal to `target` in `ref`."""
    t = normalize(target)
    hits = [i for i, w in enumerate(tokens(ref)) if w == t]
    return hits[occurrence] if len(hits) > occurrence else None


def classify_target(ops: list[Op], target_idx: int | None, group: list[str]) -> tuple[str, str | None]:
    """Status of the reference word at `target_idx` (+ the word the ASR put there)."""
    if target_idx is None:
        return MISSING, None
    members = {normalize(w) for w in group}
    pos = next(k for k, o in enumerate(ops) if o.ref_idx == target_idx)
    o = ops[pos]
    if o.op == "equal":
        return CORRECT, o.hyp
    if o.op == "substitute":
        return (HOMOPHONE if o.hyp in members else OTHER), o.hyp
    # deleted: if the aligner put a group partner as an insertion right next to it, count it as confusion
    for k in (pos - 1, pos + 1):
        if 0 <= k < len(ops) and ops[k].op == "insert" and ops[k].hyp in members:
            return HOMOPHONE, ops[k].hyp
    return DELETED, None


def analyze_sentence(ref: str, hyp: str, target: str, group: list[str], occurrence: int = 0) -> dict:
    ops, counts = align(ref, hyp)
    tidx = find_target(ref, target, occurrence)
    status, recognized = classify_target(ops, tidx, group)
    return {**counts, "ref_norm": normalize(ref), "hyp_norm": normalize(hyp), "target": target,
            "target_idx": tidx, "recognized": recognized, "status": status,
            "ops": [asdict(o) for o in ops]}


def analyze_word(target: str, hyp: str, group: list[str]) -> dict:
    """Isolated word: ALL group members are acceptable; within-group confusion measured separately."""
    h = tokens(hyp)
    members = {normalize(w) for w in group}
    t = normalize(target)
    if h == [t]:
        status = EXACT
    elif len(h) == 1 and h[0] in members:
        status = WITHIN_GROUP
    else:
        status = OTHER
    return {"target": target, "hyp_norm": " ".join(h), "status": status,
            "accepted": status in (EXACT, WITHIN_GROUP)}


def diff_spans(ops: list[dict] | list[Op], target_idx: int | None = None, status: str | None = None,
               recognized: str | None = None) -> list[tuple[str, str | None]]:
    """(text, label) pairs for a highlighted diff. Labels: None, sub, del, ins, homophone."""
    out = []
    for o in ops:
        o = o if isinstance(o, dict) else asdict(o)
        is_t = target_idx is not None and o["ref_idx"] == target_idx
        if o["op"] == "equal":
            out.append((o["hyp"], "target ok" if is_t else None))
            out.append((" ", None))
            continue
        elif o["op"] == "substitute":
            lab = "homophone" if is_t and status == HOMOPHONE else "sub"
            out.append((f"{o['ref']}→{o['hyp']}", lab))
        elif o["op"] == "delete":
            out.append((f"[-{o['ref']}]", "del"))
        else:
            # deleted target + adjacent partner insertion (see classify_target)
            hom = status == HOMOPHONE and o["hyp"] == recognized
            out.append((f"[+{o['hyp']}]", "homophone" if hom else "ins"))
        out.append((" ", None))
    return out[:-1] if out else out


def aggregate(rows: list[dict], by: tuple[str, ...] = ("accent", "cfg_weight")):
    """Per-condition metrics. WER is corpus-level (sum errors / sum reference words)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[*by, "n", "wer", "homophone_confusion_rate",
                                     "target_accuracy", "other_error_rate"])
    g = df.groupby(list(by), sort=True)
    res = g.agg(n=("status", "size"), errors=("errors", "sum"), n_ref=("n_ref", "sum"),
                conf=("status", lambda s: (s == HOMOPHONE).mean()),
                acc=("status", lambda s: (s == CORRECT).mean())).reset_index()
    res["wer"] = res["errors"] / res["n_ref"].clip(lower=1)
    res["homophone_confusion_rate"] = res.pop("conf")
    res["target_accuracy"] = res.pop("acc")
    res["other_error_rate"] = (1 - res["target_accuracy"] - res["homophone_confusion_rate"]).clip(lower=0)
    return res.drop(columns=["errors", "n_ref"]).round(4)


def aggregate_words(rows: list[dict], by: tuple[str, ...] = ("accent", "cfg_weight")):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[*by, "n", "accepted_rate", "within_group_rate", "exact_rate"])
    g = df.groupby(list(by), sort=True)
    return g.agg(n=("status", "size"), accepted_rate=("accepted", "mean"),
                 within_group_rate=("status", lambda s: (s == WITHIN_GROUP).mean()),
                 exact_rate=("status", lambda s: (s == EXACT).mean())).reset_index().round(4)


def confusion_directions(rows: list[dict]) -> dict[str, int]:
    """Counts of 'target -> recognized' for homophone confusions (shows frequency bias)."""
    out: dict[str, int] = {}
    for r in rows:
        if r.get("status") in (HOMOPHONE, WITHIN_GROUP):
            k = f"{normalize(r['target'])} → {r.get('recognized') or r.get('hyp_norm')}"
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
