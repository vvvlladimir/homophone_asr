# Smoke test – real models (CPU)

**Setup.** 5 homophone pairs (p01–p05: mehr/Meer, Wal/Wahl, Lied/Lid, Seite/Saite, Rad/Rat) × 4 sentences each = 20 sentences per voice,
plus the 10 isolated words. Voices: `tr`, `ru` (synthetic accents) and `de` (native German reference clip = control), all at `cfg_weight=0.5`.
TTS: Chatterbox Multilingual 0.1.7 (seed 1234, temperature 0.8, exaggeration 0.5). ASR: faster-whisper `small`, int8, beam 5, temperature 0.
Reference clips: one male Common Voice 17.0 speaker per language (`refs/*/SOURCE.md`). Runtime: 37 min on 4 CPU cores
(≈25–60 s per clip). Raw rows: `results/smoke/rows.json`; command:

```
homophone-asr run --accents de,tr,ru --cfg 0.5 --limit-pairs 5 --mode both --name smoke
```

> Synthetic accents are **not validated** against real L2 speech. n = 20 sentences per voice: one item = 5 percentage points.
> **No difference between voices below is statistically meaningful.**

## Context sentences

| voice | n | WER (errors/words) | homophone confusion | target correct | target → other word | target deleted |
|---|---|---|---|---|---|---|
| de | 20 | 14/115 = 12.2% | 3 | 14 | 3 | 0 |
| tr | 20 | 8/115 = 7.0% | 4 | 15 | 1 | 0 |
| ru | 20 | 23/115 = 20.0% | 3 | 12 | 5 | 0 |

Homophone confusions (target replaced by its group partner):

| voice | reference | ASR |
|---|---|---|
| de | Mein linkes Lid zuckt ständig. | Mein linkes Lied zuckt ständig. |
| de | An der Gitarre ist eine Saite gerissen. | An der Gitarbe ist eine Seite gerissen. |
| de | Die tiefe Saite klingt verstimmt. | Die tiefe Seite klingt verstimmt. |
| tr | Im Ozean schwimmt ein großer Wal. | Im Ozean schwimmt ein großer Wahl. |
| tr | Mein linkes Lid zuckt ständig. | Mein linkes Lied zuckt ständig. |
| tr | An der Gitarre ist eine Saite gerissen. | An der Gitarre ist eine Seite gerissen. |
| tr | Die tiefe Saite klingt verstimmt. | Die tiefe Seite klingt verstimmt. |
| ru | Ein Insekt ist unter mein Lid geraten. | Ein Insekt ist unter mein Lied geraten. |
| ru | An der Gitarre ist eine Saite gerissen. | und der Gitarre ist an den Seite gerissen. |
| ru | Die tiefe Saite klingt verstimmt. | Die tiefe Seite klingt verstimmt. |

Target replaced by a non-homophone (often a near-homophone: Wal→Wall/Wald, Lid→Litt, Meer→mir):

| voice | reference | ASR |
|---|---|---|
| de | Im Ozean schwimmt ein großer Wal. | Im Ozean schwimmt ein großer Wall. |
| de | Der Wal frisst winzige Krebse. | der Wallfrist winzige Klebse. |
| de | Ein Insekt ist unter mein Lid geraten. | Ein Insekt ist unter mein Litt geharten. |
| tr | Ein Insekt ist unter mein Lid geraten. | Ein Insekt ist unter mein Litt geraten. |
| ru | Das Meer ist heute sehr ruhig. | Das mir ist heute sehr ruhig. |
| ru | Im Ozean schwimmt ein großer Wal. | Im Ozean schwimmt ein großer Wall. |
| ru | Der Wal frisst winzige Krebse. | Der Wald frisst minzige Krebse. |
| ru | Das Lied läuft ständig im Radio. | das nicht löscht ständig im Radio. |
| ru | Mein linkes Lid zuckt ständig. | Man je gesteht zuständig. |

### Observations

1. **Saite → Seite in 6/6 cases, including the native control.** All confusions go from the rarer to the more frequent word
   (Saite→Seite ×6, Lid→Lied ×3, Wal→Wahl ×1). Context ("Gitarre", "verstimmt") did not stop Whisper-small from choosing the frequent word.
   This is consistent with a language-model frequency bias (not tested). The reverse direction never happened.
2. **The native control is not clean.** `de` has a homophone-confusion rate of 15 % and a WER of 12 % (e.g. "Die Fäden im Sommer…", "Gitarbe", "Klebse").
   So part of the errors comes from **TTS artifacts / mispronunciations**, not from the accent. In this run, without a listening check,
   TTS errors and ASR errors cannot be separated.
3. **tr vs. ru vs. de.** `ru` has the highest WER (20 %). `tr` has the lowest WER (7 %) but the most homophone confusions (4/20).
   With n = 20 and one speaker per voice, these differences are within noise and are confounded with speaker identity.
4. **Near-homophones are frequent.** Wal→Wall, Lid→Litt, Meer→mir. They count as `other_substitution`, but they are the same
   phenomenon (vowel length / final consonant). A future version could add a "near-homophone" class (IPA edit distance 1).

## Isolated words – FAILED (not usable in this configuration)

| voice | word | ASR | status |
|---|---|---|---|
| de | mehr | mehr. | exact |
| de | Meer | Vor blink, kennt Arbeit, kennt Arbeit, Punktung, kennt Arbei… | other_substitution |
| de | Wal | Ball. | other_substitution |
| de | Wahl | Bye. | other_substitution |
| de | Lied | Lead. | other_substitution |
| de | Lid | lit | other_substitution |
| de | Seite | Saito | other_substitution |
| de | Saite | Ka. Zatö. | other_substitution |
| de | Rad | Gott. | other_substitution |
| de | Rat | Gott. | other_substitution |
| tr | mehr | Mia. | other_substitution |
| tr | Meer | Mia? Mia? | other_substitution |
| tr | Wal | Ball. | other_substitution |
| tr | Wahl | Bye! | other_substitution |
| tr | Lied | Lead. | other_substitution |
| tr | Lid | Lead | other_substitution |
| tr | Seite | Seite, Seite. | other_substitution |
| tr | Saite | SETTA | other_substitution |
| tr | Rad | Rat und Schuhen. | other_substitution |
| tr | Rat | Rath? | other_substitution |
| ru | mehr | mehr, mehr, mehr, mehr, mehr, mehr, mehr, mehr, mehr, mehr, … | other_substitution |
| ru | Meer | Mäh, mäh, mäh. | other_substitution |
| ru | Wal | Paul. | other_substitution |
| ru | Wahl | Boah. | other_substitution |
| ru | Lied | NEED | other_substitution |
| ru | Lid | LEAD | other_substitution |
| ru | Seite | Seite, Seite. | other_substitution |
| ru | Saite | Seite Pote, Pote, fau, mi, mi, na. | other_substitution |
| ru | Rad | Das war's. | other_substitution |
| ru | Rat | Was? | other_substitution |

**Failure analysis.** Accepted rate is 1/30. The cause is the setup, not the homophones:
- Chatterbox is unstable on one-word input. It produced 0.6–0.9 s fragments, or long babble (de "Meer": 5.3 s of nonsense; "Saite": 6–7 s).
- On sub-second clips Whisper-small drifts into English despite `language="de"` ("Lead", "Bye", "Ball", "NEED").
  It also hallucinated once, a 100× repetition loop ("mehr, mehr, … more, more").

→ The word-level numbers say nothing about homophone perception. Suggested fix (not done): synthesize words in a fixed neutral carrier phrase
(e.g. "Ich sage jetzt: X.") and score only the target slot; set `compression_ratio_threshold`/`no_repeat_ngram_size` in the ASR to stop loops.

## Detector on the real ASR transcripts

The gbert detector (threshold 2.0) was run on the actual Whisper outputs. It caught **4/10** homophone confusions,
with the correct word suggested. It raised **0/41** false alarms on the target word in correctly recognized sentences.

| voice | ASR transcript | suggestion | margin | caught |
|---|---|---|---|---|
| de | Mein linkes Lied zuckt ständig. | Lid | +5.86 | ✅ |
| de | An der Gitarbe ist eine Seite gerissen. | Saite | -0.58 | ❌ |
| de | Die tiefe Seite klingt verstimmt. | Saite | +0.45 | ❌ |
| tr | Im Ozean schwimmt ein großer Wahl. | Wal | +13.35 | ✅ |
| tr | Mein linkes Lied zuckt ständig. | Lid | +5.86 | ✅ |
| tr | An der Gitarre ist eine Seite gerissen. | Saite | -0.25 | ❌ |
| tr | Die tiefe Seite klingt verstimmt. | Saite | +0.45 | ❌ |
| ru | Ein Insekt ist unter mein Lied geraten. | Lid | +2.94 | ✅ |
| ru | und der Gitarre ist an den Seite gerissen. | Saite | -4.07 | ❌ |
| ru | Die tiefe Seite klingt verstimmt. | Saite | +0.45 | ❌ |

All 6 misses are Seite/Saite: gbert also prefers the frequent *Seite*, even next to "Gitarre"/"verstimmt"
(the same 2 sentences are the misses in `detector_eval.md`). So the detector and the ASR share this blind spot. Its errors are not
independent of the ASR's.

## What worked

- The end-to-end pipeline ran with real models on CPU without crashes (90/90 items, 0 failed items).
- Reproducibility: seeded Chatterbox generation gave bit-identical audio across runs. ASR decoding is deterministic (beam 5, T=0).
- All 5 reference clips were downloaded automatically from Common Voice. The "no reference" fallback was not needed.

## Not covered by this smoke test

- `cfg_weight` sweep (only 0.5 was run). `ar`/`en` accents. Pairs p06–p12. Larger ASR models (`medium`, `large-v3`).
