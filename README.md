# homophone_asr

German homophones × synthetic L1-style accents × ASR. The pipeline takes a German sentence list, builds homophone groups, synthesizes the sentences with accented TTS, transcribes them, analyzes the errors, and runs a context-based homophone detector. Everything runs locally on CPU. A GPU is optional.

> **Synthetic accents are not validated.** The "accent" comes from cross-lingual voice prompting: German text plus a reference clip in another language. Nobody has checked that it sounds like real Turkish, Russian, Arabic or English learners of German. Read every number as a stress test of the ASR, not as a measurement of how real L2 speakers are recognized.

## Quickstart (10 lines)

```bash
sudo apt-get install -y espeak-ng ffmpeg                      # G2P backend for phonemizer
uv venv .venv --python 3.11 && source .venv/bin/activate       # chatterbox-tts 0.1.7 needs Python <3.13 + numpy<2
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    "torch==2.6.0+cpu" "torchaudio==2.6.0+cpu" -e ".[real,ui,dev]"
pytest                                                         # all tests use Mock backends: no GPU, no downloads
python app.py                                                  # demo on http://127.0.0.1:7860 (first start downloads ~3 GB of models)
python app.py --mock                                           # instant UI with simulated backends
homophone-asr run --accents tr,ru,en --cfg 0,0.3,0.5           # full experiment -> results/<name>/
homophone-asr detect "Ich will Meer Wasser."                   # detector on any text
homophone-asr detect-eval && homophone-asr groups              # detector benchmark / list homophone groups
```

## Architecture

```
 data/sentences_de.csv ─┐            data/homophone_words_de.txt
 (48 minimal-pair ctx)  │                       │
                        ▼                       ▼
                 ┌────────────── confusion.py ──────────────┐
                 │ phonemizer + espeak-ng (de) -> IPA        │  same IPA (stress stripped) = one group
                 │ mehr|Meer  Rad|Rat  das|dass  Stadt|statt │  G2P: EspeakG2P | MockG2P
                 └──────────────┬────────────────────────────┘
                                │ Lexicon
  refs/<lang>/*.wav (5-10 s) ◄── refs.py ◄── Common Voice 17 (CC0) | missing -> default voice + WARNING
          │
          ▼                     pipeline.py  (content-hash cache: .cache/homophone_asr/{audio,asr})
 ┌──── tts.py ─────┐  wav  ┌──── asr.py ──────┐ text ┌──── analyze.py ─────────────────────────┐
 │ ChatterboxTTS   │──────►│ FasterWhisperASR │─────►│ normalize -> jiwer.process_words        │
 │ lang=de + ref   │       │ beam=5, T=0      │      │ target: correct | homophone_confusion | │
 │ cfg_weight=knob │       │ | MockASR        │      │         other_substitution | deleted    │
 │ | MockTTS       │       └──────────────────┘      │ WER, confusion rate, directions         │
 └─────────────────┘                                 └──────────────┬──────────────────────────┘
                                                                    ▼
                       detector.py: gbert-base masked LM, word-level PLL(original) vs PLL(partners)
                       flag if log P(partner) − log P(original) ≥ threshold      | MockLM
                                                                    ▼
                  cli.py (run / try / detect / detect-eval / groups / fetch-refs)   app.py (Gradio, 3 tabs)
```

| Component | Choice | Notes |
|---|---|---|
| Homophone groups | `phonemizer` + espeak-ng `de` | Identical IPA after stress removal = one group. Case variants (Weise/weise) count as one spelling. Groups are computed from the data, not hand-coded. A test checks that all 12 declared pairs really share their IPA. |
| TTS | Chatterbox Multilingual, `language_id="de"` | Accent = reference clip in another language. `cfg_weight` is the accent-strength knob, from 0.0 (suppressed) to 0.5. Generation is seeded, and the same input gives a bit-identical wav on CPU. |
| Accent knobs | UI + `config.yaml` | (1) `cfg_weight` from 0 to 1.0. 0.5 is the Chatterbox default; values above 0.5 are experimental and should give a stronger accent with less clarity. (2) "Pronunciation rules": `read_as: de` (default) reads the text with German rules. `read_as: accent` passes the accent language as `language_id` (e.g. `tr`), so the German text is read with Turkish rules. This gives a much stronger but uncontrolled accent. The native `de` and `none` voices always use German rules. CLI: `--read-as accent`. |
| Reference voices | `refs/<lang>/` | One 5–10 s clip per language from Common Voice 17.0 (CC0), all male, with no accent or impairment tags. `de` is a native German control. If a clip is missing, the run uses the default voice and the UI shows a warning. |
| ASR | `faster-whisper` | Default `small`, int8 on CPU. Set `asr.model: large-v3` in `config.yaml` or pass `--asr-model`. `beam_size=5`, `temperature=0`, no previous-text conditioning, so results are reproducible. |
| Metrics | `jiwer.process_words` | Text is normalized first: NFC, lowercase (ß kept), punctuation stripped. WER is corpus-level. |
| Detector | `deepset/gbert-base` | Each candidate is scored by word-level pseudo-log-likelihood with the l2r fix (Kauf & Ivanova 2023), so multi-subword words are compared fairly. Threshold: 2.0 nats. |
| UI | Gradio | Tabs: *Try it*, *Experiment*, *Detector*. Warm-up preloads the models and the preset sentence. |

**Evaluation design.**
- **Context sentences are the real test.** The dataset covers 12 pairs, and each word appears in 2 disambiguating sentences, for 48 sentences in total. Each row reports one of: `correct`, `homophone_confusion` (ASR wrote a group partner), `other_substitution`, or `deleted`.
- **Isolated words are acoustically ambiguous.** So in `--mode words` any group member counts as correct (`accepted_rate`). Confusion within the group (`within_group_rate`) is reported separately and is not counted as an error.
- **Confusion directions** (e.g. `meer → mehr`) show the language-model frequency bias of the ASR.

**Caching and reproducibility.**
- Audio is cached under a key built from: text, the SHA-256 of the reference clip, `cfg_weight`, and the TTS parameters (model version, seed, temperature, exaggeration).
- ASR output is cached under a key built from the audio hash and all decoding parameters.
- All settings live in `config.yaml`.

## Results (small, see the files for every item)

- **`results/smoke.md`**: real-model smoke test. Chatterbox + whisper-small on CPU; 5 pairs × 4 sentences; voices tr, ru and a native-German control; `cfg_weight=0.5`.
  - **Homophone confusion: de 15 %, tr 20 %, ru 15 %. WER: de 12 %, tr 7 %, ru 20 %.**
  - Every confusion went from the rarer to the more frequent word: Saite→Seite ×6, Lid→Lied ×3, Wal→Wahl ×1.
  - The native control also has errors, so part of them is TTS artifacts. With n = 20 per voice, **no accent difference is meaningful**.
  - **Isolated-word mode failed** (1/30 accepted). Chatterbox produces fragments or babble for one-word input, and Whisper drifts into English on sub-second clips. A carrier phrase is the suggested fix (not done).
- **`results/detector_eval.md`**: on the 48 sentences, the detector flags 45/48 target→partner swaps with the correct suggestion, and raises 0/48 false alarms on the originals.
  - On the real smoke transcripts it caught 4/10 confusions, with 0/41 false alarms.
  - Every miss is Seite/Saite: gbert shares Whisper's preference for the frequent word.

## What is validated vs. not

| Validated (tested or measured here) | NOT validated |
|---|---|
| Homophone grouping: espeak-ng IPA identity, checked by tests | That synthetic accents resemble real L1-specific L2 German (no listening test, no IFCASL calibration) |
| Alignment, classification and aggregation logic (unit tests, edge cases: empty input, umlauts/ß, offsets) | That `cfg_weight` maps linearly or monotonically to perceived accent strength |
| Cache keys and reproducibility (same input = same output; Chatterbox seeded output is bit-identical on CPU) | Speaker effects: one reference speaker per language, so accent and voice are confounded |
| Detector on the 48 sentences: originals vs. target→partner swaps (`detect-eval`) | Detector threshold generality: 2.0 nats was set a priori, and the eval set is small and self-made |
| Pipeline end to end with real models (smoke test, 90/90 items, no crashes) | Statistical significance: the smoke test is tiny, so do not read differences between accents as findings |
| | Isolated-word mode with the current TTS/ASR (fails, see smoke.md) |

Mock backends (`MockTTS`, `MockASR`, `MockLM`, `MockG2P`) only exist for tests and offline UI demos. Their numbers are simulated, and the UI and reports label them **MOCK**.

## Licenses and data

- **Chatterbox** (Resemble AI): MIT. Outputs carry Resemble's imperceptible Perth watermark.
- **faster-whisper**: MIT. Whisper weights: MIT.
- **deepset/gbert-base**: MIT.
- **espeak-ng**: GPL-3.0. It is called as an external program through `phonemizer` (GPL-3.0) and is not bundled.
- **Common Voice 17.0** reference clips in `refs/`: CC0-1.0. They were downloaded through the public parquet mirror `fixie-ai/common_voice_17_0`, and the clip IDs and transcripts are listed in `refs/<lang>/SOURCE.md`. Do not try to identify the speakers.
- **IFCASL** (French/German L1-L2 corpus) and **CEASR** (corpus of ASR errors) are **not bundled** and were not used. They are listed as future calibration data: for checking synthetic accents against real L2 speech, and real ASR error patterns against the ones simulated here.

## Layout

```
homophone_asr/  confusion.py tts.py asr.py analyze.py detector.py pipeline.py cli.py refs.py cache.py config.py
app.py  config.yaml  data/  refs/  results/  tests/
```
