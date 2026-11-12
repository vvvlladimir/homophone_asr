# Detector evaluation (deepset/gbert-base)

Setup: for each of the 48 dataset sentences, (a) the original sentence must NOT be flagged at the target word; (b) the sentence with the target swapped for its homophone partner (e.g. *Das mehr ist heute sehr ruhig.*) MUST be flagged and the original target suggested. Threshold set a priori to 2.0 nats (not tuned on this set).

- detection rate (swaps flagged with correct suggestion): **93.8%** (45/48)
- false-alarm rate on originals: **0.0%** (0/48)
- threshold: 2.0 nats; skipped items: 0

## Misses

- swapped: *An der Gitarre ist eine Seite gerissen.* → best partner `Saite`, margin -0.25
- swapped: *Die tiefe Seite klingt verstimmt.* → best partner `Saite`, margin +0.45
- swapped: *Wir malen den Kaffee frisch.* → best partner `mahlen`, margin +1.06

## Margin distribution

| kind | min | median | max |
|---|---|---|---|
| original | -20.89 | -9.66 | +0.25 |
| swapped | -0.25 | +9.66 | +20.89 |

Caveat: small, self-constructed sentence set with clear disambiguating context; real ASR output (errors elsewhere in the sentence) will be harder.
