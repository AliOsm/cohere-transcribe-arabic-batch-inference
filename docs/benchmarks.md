# Accuracy Benchmarks

This page separates recognition accuracy, implementation stability, timestamp agreement, and runtime performance. Those are different claims: matching a stored transcript shows that an implementation change preserved the established output for that input and configuration, WER and CER require authoritative text references, and timestamp accuracy requires human boundary annotations.

Word error rate (WER) and character error rate (CER) are lower-is-better edit rates. `pp` means percentage points, and a clip is one presegmented row in the frozen evaluation manifest.

See [Performance](performance.md) for v0.1.0 installed-wheel timing, resource usage, decoder and VAD throughput, alternate engines, and the reasons behind the runtime defaults.

## Accuracy Summary

The largest retained local evaluation is a frozen 24,414-clip Arabic suite containing 36.393 decoded hours. It predates the packaged v0.1.0 wheel and evaluates the same optimized Cohere ASR path without VAD or alignment. On its primary lexical-normalized scoring profile, it measured 31.3205% corpus WER and 14.2408% corpus CER.

| Scope | Clips | Decoded hours | WER | CER | WER substitutions / deletions / insertions |
|---|---:|---:|---:|---:|---:|
| Overall | 24,414 | 36.393 | 31.3205% | 14.2408% | 46,576 / 17,756 / 7,629 |
| MSA | 11,056 | 14.499 | 6.1730% | 1.9516% | 3,155 / 581 / 289 |
| Dialect | 12,758 | 17.912 | 42.5972% | 19.7561% | 43,169 / 17,125 / 6,426 |
| Classical Arabic recitation proxy | 600 | 3.982 | 15.3458% | 11.0651% | 252 / 50 / 914 |

These are corpus-micro rates over presegmented clips. The MSA group contains 10,471 Common Voice clips, 428 FLEURS clips, and 157 SADA clips; the dialect group contains 6,726 Casablanca clips and the other 6,032 SADA clips. They are not a long-form VAD benchmark, they are not the official Cohere leaderboard average, and the Classical Arabic row is a limited Quran recitation proxy rather than a general Classical Arabic score.

On Common Voice, SADA, and Casablanca, the local optimized inference code closely reproduces Cohere's officially reported results on the exact overlapping leaderboard rows. Local WER is within 1.33 percentage points and local CER is within 3.54 percentage points of the official value on every overlapping dataset across the two documented local normalization profiles.

## Frozen Evaluation Suite

| Dataset | Role | Clips | Decoded hours | Pinned revision |
|---|---|---:|---:|---|
| [Common Voice 18 Arabic](https://huggingface.co/datasets/MohamedRashad/common-voice-18-arabic/tree/1a52eefd8259398b1ddda495876f71c202943df2) | Arabic read speech; local MSA grouping | 10,471 | 12.657 | `1a52eefd8259398b1ddda495876f71c202943df2` |
| [FLEURS `ar_eg`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) | Arabic `ar_eg` parallel read speech; local MSA grouping | 428 | 1.302 | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |
| [SADA22](https://huggingface.co/datasets/MohamedRashad/SADA22/tree/094fe2c0fe4b549a4f34349e6e0622e7c7273c2d) | Saudi TV broadcast speech, chiefly Saudi plus other Arabic varieties | 6,189 | 10.749 | `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d` |
| [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca/tree/8951b1b88e28c1107142ced57967b8d16350951d) | TV-series dialogue across eight country varieties | 6,726 | 7.703 | `8951b1b88e28c1107142ced57967b8d16350951d` |
| [Quran Ayah Corpus sample](https://huggingface.co/datasets/rabah2026/Quran-Ayah-Corpus/tree/80cad1ab411c2e54ba3411553c8bb7fab5f47042) | Classical Arabic recitation proxy | 600 | 3.982 | `80cad1ab411c2e54ba3411553c8bb7fab5f47042` |

Total decoded duration is 131,015.816 seconds, or 36.393282 hours. Source metadata totals 36.301534 hours; the 330.292-second difference is almost entirely Common Voice MP3 duration reporting. WER and CER are unaffected by which duration denominator is used.

Common Voice, SADA22, and Casablanca IDs and references were matched to the exact evaluation rows in the [Open Universal Arabic ASR Leaderboard at commit `10cf2c8`](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/tree/10cf2c8f257c26467f6d9210b28d584d2fec6a2e). The Common Voice source is an unofficial Arabic-only extraction that preserves the Mozilla CV18 splits. The 6,189 SADA and 6,726 Casablanca rows are frozen leaderboard subsets, not their source-card test totals of 6,193 and 6,818. The Quran selection consists of 12 evenly spaced blocks of 50 rows from a held-out split containing only three reciters. Its references are canonical verse text, not independently produced verbatim transcripts.

`MSA` is an evaluation grouping, not proof that every speaker uses MSA or the same accent. It contains all Common Voice and FLEURS clips plus 157 SADA clips labelled MSA. The FLEURS `ar_eg` label identifies the locale/configuration, not Egyptian colloquial speech. The dialect group combines all Casablanca clips with the other 6,032 SADA clips. SADA contributes 2,249 rows labelled unknown, not applicable, or multi-speaker, so the aggregate dialect result is more defensible than ranking every source label individually.

Dataset terms also differ: Common Voice is CC0-1.0 under the [Mozilla terms](https://commonvoice.mozilla.org/terms), FLEURS is CC-BY-4.0, SADA is CC-BY-NC-SA-4.0, and Casablanca is CC-BY-NC-ND-4.0 with original-media rights retained. The Quran source is unresolved: its card metadata says Apache-2.0, its prose says CC-BY-NC-SA-4.0, and the [upstream recitation terms](https://alquran.cloud/terms-and-conditions) retain reciter or estate restrictions. Check the primary sources before redistributing evaluation audio.

Frozen manifest provenance:

```text
manifest: benchmark/manifests/wer_eval.jsonl
samples: 24,414
manifest SHA-256: 332aa2a063cf285a584bc1630f9164c15d18ea904b79039c39585cf9919a277d
ID/reference SHA-256: 2e8108d2a02cbd049d1c722bebc994e05848d10c3d4bcb19270403764f339395
```

The manifest belongs to the retained research workspace and is not included in this release tree, source distribution, or wheel. Its hashes are included so a future rerun with access to that workspace can prove that it used the same samples and references.

## Accuracy Method

The full suite used `CohereLabs/cohere-transcribe-arabic-07-2026` at revision `0a8193caa4f3f92131471ab08824e488141cb392`, BF16, PyTorch SDPA, duration-sorted batch 24, a 445-token limit, the encoder projection cache, and the conservative repetition-loop stop. The configuration fingerprint is `23573682e9d5f8b7fddb7e1d77c9b56993eb483d09b5ebca3070e65b70832710`.

The benchmark presents one already segmented utterance to ASR for each manifest row. It deliberately excludes Silero boundary selection, subtitle construction, and forced alignment. This isolates recognition behavior, but it does not estimate accuracy on an unsegmented recording.

WER and CER are accumulated from corpus-level substitution, deletion, insertion, and reference-unit counts rather than averaging per-clip percentages. CER operates on the normalized character stream, including normalized spaces. Equal-cost Levenshtein paths use Kaldi/NeMo-compatible insertion, deletion, then substitution tie-breaking; this changes the S/D/I breakdown but not the total edit distance.

Paired comparisons use 5,000 percentile-bootstrap replicates at 95% confidence with seed 0. Common Voice speakers and the three Quran reciters are resampled as dataset-namespaced speaker clusters; rows without usable speaker IDs are resampled by utterance. The overall comparison contains 14,319 clusters: 976 speaker clusters and 13,343 utterance clusters. A confidence interval that excludes zero supports a difference on this frozen suite, not universal generalization.

## Text Normalization

The headline profile is `lexical_normalized`. It applies Unicode NFKC and case folding, removes every Unicode combining mark and tatweel, maps punctuation to spaces, and collapses whitespace. It folds `پ` to `ب`, `ڤ` to `ف`, `آ`/`أ`/`إ`/`ٱ` to `ا`, `ؤ` to `و`, `ئ`/`ى`/Persian `ی` to `ي`, Persian `ک` to `ك`, removes standalone `ء`, and converts Eastern Arabic digits to ASCII. Ta marbuta `ة` is preserved. This is the closest local implementation of the normalization disclosed in Cohere's [technical release](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release#normalization).

| Profile | Purpose |
|---|---|
| `lexical_normalized` | Headline WER/CER and the closest available implementation of Cohere's disclosed Arabic normalization |
| `leaderboard_intended` | The public leaderboard's described punctuation removal followed by its Arabic diacritic, hamza, and digit mappings |
| `leaderboard_repo_exact` | Exact reproduction of evaluator commit `10cf2c8`, including its literal punctuation-regex behavior; audit only |
| `raw` | Exact output spelling, punctuation, and diacritics; WER splits on whitespace while CER preserves literal spaces |

All 24,414 references remain nonempty lexical token streams. The local lexical normalizer is an independent implementation of the disclosed behavior, not a verified byte-identical copy of Cohere's internal Whisper-derived normalizer. Headline rates from different normalizers must not be compared as though they used the same scorer.

The overall normalization sensitivity is substantial:

| Profile | Cohere WER | Cohere CER | Wit/Tafrigh WER | Wit/Tafrigh CER |
|---|---:|---:|---:|---:|
| Raw | 49.0929% | 21.7634% | 48.8504% | 26.2378% |
| Leaderboard repository exact | 38.0304% | 16.5932% | 38.0369% | 21.1319% |
| Leaderboard intended | 31.5924% | 14.4466% | 34.1945% | 20.1743% |
| Lexical normalized | 31.3205% | 14.2408% | 34.0011% | 20.0255% |

## Cohere Results

### By Dataset

| Dataset | Clips | WER | CER | WER substitutions / deletions / insertions |
|---|---:|---:|---:|---:|
| Common Voice 18 Arabic | 10,471 | 5.5437% | 1.5336% | 2,495 / 275 / 184 |
| FLEURS `ar_eg` | 428 | 4.7458% | 2.1546% | 232 / 96 / 52 |
| SADA22 | 6,189 | 36.1444% | 19.9980% | 17,751 / 9,676 / 3,702 |
| Casablanca | 6,726 | 48.7556% | 18.7316% | 25,846 / 7,659 / 2,777 |
| Quran recitation proxy | 600 | 15.3458% | 11.0651% | 252 / 50 / 914 |

The overall rate is dominated by the reference-word distribution, not by an equal average of the five dataset percentages. Report dataset rows whenever deployment data resembles one source more than the full mixture.

### Repetition Guard

The selected BF16 length-sorted baseline measured 32.2576% WER. Enabling the documented periodic-loop stop with the projection optimization changed 125 of 24,414 hypotheses and reduced WER on this suite to 31.3205%, a difference of -0.9371 percentage points with paired 95% CI `[-1.4766, -0.4781]`.

| Scope | Baseline WER | Guarded WER | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Overall | 32.2576% | 31.3205% | -0.9371 pp | `[-1.4766, -0.4781]` |
| MSA | 6.1730% | 6.1730% | 0.0000 pp | `[0.0000, 0.0000]` |
| Dialect | 43.7190% | 42.5972% | -1.1218 pp | `[-1.7432, -0.5597]` |
| Classical Arabic proxy | 20.3433% | 15.3458% | -4.9975 pp | `[-13.1912, 0.0000]` |

The change primarily removed long insertion loops. The Classical Arabic estimate has only three resampling clusters, five changed hypotheses, and a confidence interval touching zero; it is not evidence of a general five-point Quran improvement.

## Cohere and Wit.ai Through Tafrigh

The matched comparison uses the same 24,414 IDs, references, metadata, and lexical scorer. The lexical references contain 229,757 words. The Wit run was collected on July 10, 2026 through [Tafrigh-compatible preprocessing at revision `2ccba42`](https://github.com/ieasybooks/tafrigh/tree/2ccba42db8c34c04924d1befc35cb3d5eec80d93) with eight distinct Arabic apps. Its configuration fingerprint is `8a1df34ae555df7e62eac2cc448f6ccade36f44fc90e0dcc31e125b160112aae`.

All 24,414 clips have paired Cohere and Wit hypotheses. Overall Cohere WER substitutions/deletions/insertions are `46,576 / 17,756 / 7,629`; the corresponding Wit counts are `34,499 / 38,952 / 4,669`. Wit produces fewer substitutions and insertions but substantially more deletions.

This is an end-to-end system comparison, not a model-only comparison. Cohere consumes each frozen utterance directly. Tafrigh converts audio as needed, applies Auditok with a 15-second maximum region, re-encodes regions as MP3, adds one second of Tafrigh-generated noise on both sides, calls the Wit `/speech` service, and concatenates returned segments. Wit outputs also varied across repeated calls/apps. These differences are part of the measured Tafrigh system and prevent attributing every WER delta to the underlying Wit recognizer.

Every delta below is `Wit minus Cohere`; positive values favor Cohere and negative values favor Wit.

### By Domain

| Scope | Cohere WER / CER | Wit WER / CER | WER delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Overall | 31.3205% / 14.2408% | 34.0011% / 20.0255% | +2.6807 pp | `[+1.6651, +4.1626]` |
| MSA | 6.1730% / 1.9516% | 13.9610% / 7.2263% | +7.7880 pp | `[+7.1544, +8.4692]` |
| Dialect | 42.5972% / 19.7561% | 41.9543% / 24.5857% | -0.6429 pp | `[-1.1236, -0.1854]` |
| Classical Arabic proxy | 15.3458% / 11.0651% | 41.6961% / 38.4623% | +26.3503 pp | `[+8.3661, +53.1312]` |

Wit has lower aggregate dialect WER, while Cohere has lower aggregate dialect CER. That disagreement is possible because Wit produces more deletions and fewer substitutions/insertions; neither single metric fully describes transcript usefulness.

### By Dataset

| Dataset | Cohere WER / CER | Wit WER / CER | WER delta | Paired 95% CI | Result on this suite |
|---|---:|---:|---:|---:|---|
| Common Voice 18 Arabic | 5.5437% / 1.5336% | 12.8683% / 5.9052% | +7.3246 pp | `[+6.7098, +8.0219]` | Cohere lower WER |
| FLEURS `ar_eg` | 4.7458% / 2.1546% | 19.7327% / 14.6829% | +14.9869 pp | `[+12.5980, +17.5223]` | Cohere lower WER |
| SADA22 | 36.1444% / 19.9980% | 34.8857% / 21.9023% | -1.2587 pp | `[-2.0175, -0.5050]` | Wit lower WER; Cohere lower CER |
| Casablanca | 48.7556% / 18.7316% | 48.8255% / 26.7759% | +0.0699 pp | `[-0.4783, +0.5709]` | No decisive WER difference |
| Quran recitation proxy | 15.3458% / 11.0651% | 41.6961% / 38.4623% | +26.3503 pp | `[+8.3661, +53.1312]` | Cohere lower WER; only three reciters |

The aggregate ranking changes with dataset composition. The +2.68 pp overall result must not be presented as a universal ranking of Arabic recognition quality.

### By Source Variety and Annotation Label

This primary-profile breakdown excludes MSA and Classical Arabic because they already appear in the domain table. The remaining rows mix Casablanca country labels, SADA variety labels, and annotation categories. Similarly named labels such as Egypt/Egyptian and Yemen/Yemeni come from different source schemas and were not merged.

| Source label | Clips | Cohere WER / CER | Wit WER / CER | WER delta | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Algeria | 843 | 63.04% / 24.31% | 59.53% / 31.71% | -3.51 pp | `[-5.49, -1.82]` |
| Egypt | 825 | 31.61% / 11.93% | 36.69% / 15.46% | +5.07 pp | `[+3.64, +6.41]` |
| Egyptian | 96 | 35.24% / 16.74% | 30.64% / 14.54% | -4.61 pp | `[-10.26, +0.52]` |
| Hijazi | 809 | 31.99% / 15.96% | 32.90% / 20.78% | +0.91 pp | `[-0.71, +2.60]` |
| Jordan | 848 | 29.69% / 8.82% | 31.93% / 11.96% | +2.24 pp | `[+1.01, +3.42]` |
| Khaliji | 1,150 | 37.70% / 18.56% | 42.20% / 28.57% | +4.49 pp | `[+2.17, +6.64]` |
| Mauritania | 948 | 79.75% / 41.53% | 84.94% / 73.05% | +5.19 pp | `[+3.81, +6.56]` |
| Multiple speakers | 1,320 | 39.42% / 23.46% | 35.86% / 22.22% | -3.56 pp | `[-4.92, -2.32]` |
| Morocco | 1,045 | 54.50% / 17.41% | 43.01% / 17.77% | -11.50 pp | `[-12.78, -10.23]` |
| Najdi | 1,703 | 31.04% / 16.49% | 28.31% / 16.54% | -2.73 pp | `[-4.11, -1.45]` |
| Not applicable | 167 | 44.12% / 22.36% | 51.02% / 34.82% | +6.90 pp | `[+2.91, +10.93]` |
| Palestine | 667 | 37.96% / 12.41% | 36.26% / 12.87% | -1.69 pp | `[-2.86, -0.54]` |
| Shamali | 18 | 26.73% / 10.62% | 30.69% / 18.28% | +3.96 pp | `[-0.91, +12.50]` |
| UAE | 813 | 40.38% / 12.92% | 45.19% / 24.30% | +4.81 pp | `[+3.29, +6.28]` |
| Unknown | 762 | 44.40% / 25.57% | 48.54% / 36.35% | +4.14 pp | `[+0.28, +7.58]` |
| Yemen | 737 | 50.62% / 19.62% | 53.19% / 27.42% | +2.58 pp | `[+1.27, +3.88]` |
| Yemeni | 7 | 65.22% / 59.09% | 63.04% / 27.73% | -2.17 pp | `[-18.75, +9.43]` |

These labels are not a controlled dialect taxonomy. Metadata buckets and very small groups, especially Shamali and Yemeni, should not be ranked as stable dialect estimates.

### Wit/Tafrigh Completion Audit

The full run recorded 32,160 successful speech-segment responses across 32,192 HTTP attempts. All 32 retries recovered, so no exhausted transport or API failure was silently scored as empty text. Auditok found no speech in 82 clips, and 4,980 successful requests returned empty text across 3,954 clips; those are measured end-to-end system outputs and remain part of the WER/CER result.

## Official Cohere Results

The [official model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026#results) reports the following Open Universal Arabic ASR Leaderboard results as of July 7, 2026:

| Official task | Reported WER | Reported CER |
|---|---:|---:|
| Six-task macro average | 25.87% | 11.80% |
| SADA | 37.47% | 23.53% |
| Common Voice | 5.82% | 1.62% |
| MASC clean | 19.60% | 6.45% |
| MASC noisy | 27.07% | 10.13% |
| MGB-2 | 15.54% | 8.40% |
| Casablanca | 49.71% | 20.66% |

The official 25.87% is the arithmetic mean of six task WERs. The local 31.3205% headline is a corpus-micro WER over a different five-dataset suite that omits MASC and MGB-2 and adds FLEURS and the Quran proxy. Those two overall values are not comparable.

For the three overlapping datasets, both disclosed local profiles are shown because neither is known to be byte-identical to Cohere's unpublished scorer:

| Dataset | Official WER / CER | Local lexical WER / CER | Local leaderboard-intended WER / CER |
|---|---:|---:|---:|
| Common Voice | 5.82% / 1.62% | 5.5437% / 1.5336% | 5.8535% / 1.7765% |
| SADA | 37.47% / 23.53% | 36.1444% / 19.9980% | 36.2245% / 20.1095% |
| Casablanca | 49.71% / 20.66% | 48.7556% / 18.7316% | 48.8713% / 18.9731% |

Across the two disclosed normalization profiles, local WER differs from the official value by at most 1.33 percentage points and local CER by at most 3.54 percentage points on the three overlapping datasets. Exact installed-wheel output parity on the retained long-form and balanced-500 workloads supports that the packaged implementation preserves this inference path, but the complete 24,414-clip suite has not been rerun from the installed wheel.

The small remaining differences cannot be resolved exactly because Cohere's internal Whisper-derived normalizer, submitted hypotheses, and inference configuration are not public. The lexical profile follows Cohere's normalization disclosure, while `leaderboard_intended` follows the public evaluator's prose. The public evaluator's documented and implemented punctuation behavior also diverge, and the local repetition guard changes outputs that enter a documented periodic pattern.

## Configuration Sensitivity

### Batch Ordering and Size

BF16 generation is not guaranteed to produce identical text when batch composition changes. On the full suite, length sorting changed 1,418 of 24,414 hypotheses relative to ordered batch 24, but the WER delta was not statistically distinguishable from zero.

| Configuration | WER | CER | WER delta vs ordered batch 24 | Paired 95% CI |
|---|---:|---:|---:|---:|
| Ordered batch 24 | 32.3006% | 16.4288% | Baseline | Not applicable |
| Length-sorted batch 16 | 32.2841% | 16.4230% | -0.0165 pp | `[-0.2877, +0.2201]` |
| Length-sorted batch 24 | 32.2576% | 16.3856% | -0.0431 pp | `[-0.2136, +0.0809]` |
| Length-sorted batch 32 | 32.4129% | 16.5457% | +0.1123 pp | `[-0.1385, +0.3495]` |

This supports the selected length-sorted batch 24 configuration on the tested RTX 3060; it does not prove token identity across devices, dtypes, or batch schedules.

### VAD and Segment Construction

The balanced 500-file probe contains 100 already segmented utterances from each dataset and 5,035.715 seconds of audio. Because each source is already an evaluation utterance, it favors retaining the complete clip and penalizes a VAD that removes quiet or short reference speech. It is a configuration-sensitivity test, not an estimate of VAD accuracy on continuous recordings.

| Probe revision and segmentation | Rows | Lexical WER | Interpretation |
|---|---:|---:|---|
| Historical rounded-boundary Silero | 729 | 32.4073% | Earlier implementation baseline |
| Sample-exact Silero, static batch 24 | 729 | 31.2898% | Same span count, changed sample boundaries and context |
| Sample-exact Silero, adaptive batching | 729 | 31.0570% | Different BF16 batch composition; not text-identical |
| Sample-exact Silero with merge, static batch 24 | 508 | 27.6424% | More ASR context and fewer rows on presegmented clips |

A separate controlled probe of the earlier implementation measured 32.4073% for Silero, 28.4184% for Auditok, and 23.2656% for fixed no-VAD windows. The whole-clip ASR harness scored 22.9551%. Lower WER without VAD is expected on already clipped utterances and must not be extrapolated to arbitrary long audio, where silence hallucinations and boundary placement matter. Cohere's model card explicitly recommends VAD because silence can produce hallucinations.

Sample-exact Silero and merge were evaluated only on the balanced 500-file probe. The 24,414-clip ASR suite bypassed VAD entirely, so it does not answer end-to-end VAD accuracy on a full reference set. The v0.1.0 wheel reproduced the retained outputs for the 69-minute Arabic grammar lecture and all 500 balanced files, which provides implementation-regression evidence for those workloads but does not replace a full installed-wheel WER rerun.

## Timestamp Limits

Recognition text is produced before segment interpolation or CTC forced alignment. In the retained balanced-500 timing fixture, text-only, segment-timed, FP16-aligned, and FP32-aligned runs used identical segmentation and ASR inputs and matched all 6,119 words, so their WER was identical by construction and by measurement.

None of Common Voice, FLEURS, SADA22, Casablanca, or the Quran sample provides human word-boundary annotations. FP16-versus-FP32 comparisons therefore measure implementation agreement, not absolute timestamp accuracy. In retained validation of the aligner implementation packaged in v0.1.0, 99.77% of balanced-500 boundaries and 99.94% of long-form boundaries were within 20 ms of the FP32 implementation, but FP32 itself is not ground truth. Segment timing is explicitly approximate and should never be described as forced word alignment.

In the older aligner timing fixture, segment interpolation differed from FP32 by a 260 ms median and 1,331.85 ms p95 absolute boundary distance; only 12.706% of boundaries were within 20 ms. FP16 word alignment differed by a 0 ms median/p99 and 0.912 ms mean; 99.665% were within 20 ms, but four of 12,238 boundaries exceeded 500 ms and two exceeded one second in repetitive Quran text. These historical tails do not characterize the packaged aligner without a new full-distribution rerun. They are method-agreement values, not acoustic error against human labels.

The 69-minute Arabic grammar lecture has no human reference transcript. Matching its output hashes proves repeatability, not WER, and disagreements between its Silero, Auditok, and no-VAD transcripts are edit-distance disagreements rather than accuracy measurements.

## Applicability to v0.1.0

The full 24,414-clip inference run bypassed VAD and alignment and was not rerun from the installed wheel. It therefore characterizes the optimized Cohere ASR path and selected generation safeguards, not end-to-end v0.1.0 behavior on unsegmented recordings. Sample-exact VAD and merge behavior were evaluated separately on the balanced 500-file probe.

A built v0.1.0 wheel reproduced the retained long-form TXT, SRT, and VTT hashes and all 500 balanced-corpus transcripts. This is implementation-regression evidence, not a replacement for a full installed-wheel WER rerun. Package performance values are in [Performance](performance.md), and wheel validation evidence is in [`reports/0.1.0-release-validation.json`](../reports/0.1.0-release-validation.json).

## What the Evidence Does Not Establish

- The full-suite rates do not include long-form VAD errors, silence hallucinations, diarization, or subtitle timing.
- The Quran proxy has only three reciters and canonical verse references; it is not a general Quranic or Classical Arabic benchmark.
- The suite does not cover MASC, MGB-2, spontaneous Arabic-English code-switching, or in-domain production data.
- WER does not measure punctuation quality, semantic fidelity, named entities, dialect faithfulness, hallucination severity, or downstream task utility.
- CER and WER can rank systems differently, as seen in the dialect and SADA results.
- Wit/Tafrigh results include cloud nondeterminism, Auditok, MP3 encoding, padding, and service behavior; they are not a controlled recognizer-only comparison.
- The v0.1.0 package has exact regression parity on the retained long-form and balanced-500 workloads, but no installed-wheel full 24,414-clip rerun has been completed.
- No absolute word-timestamp error can be reported without a human-aligned timing set.

## Evidence and Provenance

The retained research workspace used the following primary machine-readable artifacts. They are not included in this release tree, source distribution, or wheel; the paths and hashes identify the frozen evidence used for this page. This ledger identifies the source artifacts but does not make the results independently reproducible without access to them.

| Artifact | SHA-256 | Purpose |
|---|---|---|
| `benchmark/manifests/wer_eval.jsonl` | `332aa2a063cf285a584bc1630f9164c15d18ea904b79039c39585cf9919a277d` | Frozen 24,414-row evaluation manifest |
| `benchmark/manifests/wer_eval.summary.json` | `fdd1e624027a0ae7fbc213e9adaab399353d349fb8d03ee4921b1f89023b9b20` | Manifest composition and duration summary |
| `benchmark/reports/final_full_20260711.json` | `bd49322844748b87d34679ac716eaefa4c0694db637b631c0d40253657ad66be` | Selected Cohere configurations and accuracy metrics |
| `benchmark/reports/final_cohere_vs_wit_20260711.json` | `c8306bcd7f70f006e271950a0cf4a052741cfed5e316ffb0a90ebc401f91a12f` | Paired Cohere/Wit profiles, group metrics, and confidence intervals |
| `benchmark/results/wit_full_default_20260710/summary.json` | `2726e15b57156298849b5315d9fff6bbca471a1983924afda653e1209d3f3c89` | Wit/Tafrigh completion and request telemetry |
| `benchmark/reports/vad_modes_500_20260711.json` | `1c1e825dd1ca688a4badbc6d7efb5bef10f8e3e10de128ebe28427bb31220237` | Balanced-500 segmentation sensitivity |
| `benchmark/reports/timestamp_modes_500_20260711.json` | `8947fb3ae5d8023d73114a66e88586143262e58eed773a6e00ce75394db094d1` | Balanced-500 timing-mode comparison |

Before accepting an accuracy claim, verify the model revision, inference fingerprint, manifest hash, scorer profile, grouping policy, sample count, and whether the run included VAD. Before accepting a performance claim, repeat complete external process timing after a warm model download and preserve transcript/reference comparisons alongside the timing result.
