# Accent-strength check (real models, 2 sentences)

Proxy for accent strength: Whisper-small **language-ID probability that the audio is German** (P(de); auto-detect, not forced).
Lower = sounds less German. The transcript below is from the normal pipeline (language forced to `de`).
Tiny sample (2 sentences, 1 speaker per voice): indicative only, not validated by listeners.

| sentence | voice | cfg_weight | pronunciation rules | P(de) | detected | ASR (forced de) | target |
|---|---|---|---|---|---|---|---|
| Die tiefe Saite klingt verstimmt. | de | 0.5 | de | 0.996 | de | Die tiefe Seite klingt verstimmt. | homophone_confusion |
| Die tiefe Saite klingt verstimmt. | de | 1.0 | de | 0.988 | de | Die tiefe Seite klingt bestimmt. | homophone_confusion |
| Die tiefe Saite klingt verstimmt. | tr | 0.5 | de | 0.997 | de | Die tiefe Seite klingt verstimmt. | homophone_confusion |
| Die tiefe Saite klingt verstimmt. | tr | 1.0 | de | 0.997 | de | Die tiefe Seite klingt verstimmt. | homophone_confusion |
| Die tiefe Saite klingt verstimmt. | tr | 0.5 | tr | 0.015 | tr | Die TFS-Seite klingt wärstend. | homophone_confusion |
| Die tiefe Saite klingt verstimmt. | tr | 1.0 | tr | 0.317 | tr | Die TFS-Seite klingt verstimmt. | homophone_confusion |
| Das Meer ist heute sehr ruhig. | de | 0.5 | de | 0.963 | de | Das Meer ist heute sehr ruhig. | correct |
| Das Meer ist heute sehr ruhig. | de | 1.0 | de | 0.912 | de | Das Meer ist heute sehr ruhig. | correct |
| Das Meer ist heute sehr ruhig. | tr | 0.5 | de | 0.988 | de | Das Meer ist heute sehr ruhig. | correct |
| Das Meer ist heute sehr ruhig. | tr | 1.0 | de | 0.996 | de | Das Meer ist heute sehr ruhig. | correct |
| Das Meer ist heute sehr ruhig. | tr | 0.5 | tr | 0.004 | en | Das Meer ist heute sehr ruhig. | correct |
| Das Meer ist heute sehr ruhig. | tr | 1.0 | tr | 0.002 | fa | Das Meer ist heute sehr ruhig. | correct |

**Reading.** Raising `cfg_weight` from 0.5 to 1.0 with German rules changes nothing measurable: the Turkish voice still gets P(de) ≈ 0.99, like the native control.
Reading the text with Turkish rules (`read_as: accent`) makes Whisper's language ID drop to P(de) 0.002–0.32 (detected as tr/en/fa).
Recognition is still sometimes perfect ("Das Meer ist heute sehr ruhig.") and sometimes breaks ("Die TFS-Seite klingt wärstend.").
Command: one-off script over `Pipeline.run_sentence(..., read_as=...)` + `faster_whisper` language detection.
