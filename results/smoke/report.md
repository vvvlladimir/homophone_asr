# smoke

- TTS: `{'backend': 'chatterbox-mtl', 'version': '0.1.7', 'lang': 'de', 'exaggeration': 0.5, 'temperature': 0.8, 'seed': 1234}`
- ASR: `{'backend': 'faster-whisper', 'model': 'small', 'compute_type': 'int8', 'language': 'de', 'beam_size': 5, 'temperature': 0.0, 'condition_on_previous_text': False, 'vad_filter': False}`
- reference clips: `{'de': 'cv17_common_voice_de_20272292.wav', 'tr': 'cv17_common_voice_tr_18967997.wav', 'ru': 'cv17_common_voice_ru_18988621.wav'}`
- runtime: 2197.3 s

Synthetic accents are produced by cross-lingual voice prompting and are NOT validated against real L2 speech.

## Sentences (context test)

| accent   |   cfg_weight |   n |    wer |   homophone_confusion_rate |   target_accuracy |   other_error_rate |
|:---------|-------------:|----:|-------:|---------------------------:|------------------:|-------------------:|
| de       |          0.5 |  20 | 0.1217 |                       0.15 |              0.7  |               0.15 |
| ru       |          0.5 |  20 | 0.2    |                       0.15 |              0.6  |               0.25 |
| tr       |          0.5 |  20 | 0.0696 |                       0.2  |              0.75 |               0.05 |

## Isolated words (all group members accepted)

| accent   |   cfg_weight |   n |   accepted_rate |   within_group_rate |   exact_rate |
|:---------|-------------:|----:|----------------:|--------------------:|-------------:|
| de       |          0.5 |  10 |             0.1 |                   0 |          0.1 |
| ru       |          0.5 |  10 |             0   |                   0 |          0   |
| tr       |          0.5 |  10 |             0   |                   0 |          0   |

## Confusion directions (target → recognized)

- saite → seite: 6
- lid → lied: 3
- wal → wahl: 1
