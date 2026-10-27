# Estimated coding scores

SlipAgent ranks eligible free endpoints using one synthetic text-coding score
per model, from 0 to 100, with higher values preferred. The score is an estimate
for comparing models across different coding benchmarks. It is not a measured
pass rate or a probability that SlipAgent will complete a task.

Only scores and exact model aliases enter runtime metadata. Free pricing, keys,
provider availability, context limits, and request failures remain separate
selection checks. A paid model can help calibrate scores without becoming an
automatic routing candidate. Visual benchmark results contribute nothing.

## Calibration reviewed October 27, 2026

The scores are computed from raw published results saved in
[`scripts/model-score-measurements.json`](../scripts/model-score-measurements.json):
218 results for 112 models, including 45 reference models that are not scored for
routing. Nine text-coding benchmarks are connected through shared models.
SWE-bench Verified supplies the reference scale. DeepSWE 1.1, Terminal-Bench 2.0
and 2.1, LiveCodeBench, HumanEval, BigCodeBench complete, FLTEval (a Lean formal-proof benchmark), and the MT-Bench coding
rating supply the rest.

For a measured percentage `p`, convert it to log-odds:

```text
logit(p / 100) = model_ability - benchmark_offset
estimated_coding_score = 100 / (1 + exp(-model_ability))
```

`scripts/update-model-scores.py --fit` fits every model ability and every benchmark
offset jointly, fixing the SWE Verified offset at zero. Huber loss with a 0.75
log-odds threshold reduces the influence of conflicting results. It keeps every
observation. Probabilities are clipped to 0.005–0.995 to handle published
zero/full scores. One decimal place is stored.

A benchmark is calibrated only when a chain of models links it to SWE Verified and
at least two models have results on it and on another benchmark. Reference models
supply those links, whether or not any free endpoint serves them. When a new model
is published on a benchmark no other model has, add the other models that report
that benchmark together with a calibrated one, and the fit places it on the scale.
The command refuses a benchmark it cannot connect.

| Benchmark | Offset | Results | Models with another result |
| --- | ---: | ---: | ---: |
| SWE Verified | 0.00 | 36 | 36 |
| DeepSWE 1.1 | +2.25 | 26 | 25 |
| Terminal 2.0 | +1.17 | 12 | 12 |
| Terminal 2.1 | +0.81 | 40 | 37 |
| LiveCodeBench | −0.61 | 27 | 23 |
| BigCodeBench complete | −1.30 | 30 | 24 |
| HumanEval | −2.60 | 35 | 23 |
| FLTEval pass@1 | +2.25 | 5 | 4 |
| MT-Bench coding | −2.30 | 7 | 6 |

The agentic benchmarks agree with the previous review, which used a different, partly
unrecorded set of results: the DeepSWE and Terminal-Bench offsets moved by 0.3 log-odds or
less, and the scores of the new-generation models moved by one to three points. The
HumanEval, BigCodeBench, and MT-Bench offsets moved by 0.6 to 0.7 toward harder. Small
and older models with LiveCodeBench or SWE Verified results alongside them (Gemma 3,
Gemini 1.5, 2.0, and 2.5) show that a HumanEval score near 85–92 does not imply a high
SWE Verified score. The older models whose only evidence is HumanEval or BigCodeBench
therefore fell by up to 30 points: Gemma 3 12B 47.3 to 21.7, Mistral Large 2 63.8 to
46.1, and Nemotron-4 340B 29.5 to about 17.

Primary sources are the models' own cards and technical reports, the
[EvalPlus](https://evalplus.github.io/results.json) and
[BigCodeBench](https://bigcode-bench.github.io/results.json) result files, and
Google's, Moonshot's, Poolside's, and Alibaba's comparison tables for reference
models. Every record keeps its source links. Several 2026 results are developer-reported
and not independently reproduced. Four are third-party quotations rather than a
developer page: Step 5 Preview (BenchLM), dots-3 note preview (BenchLM), MiniMax M3.1
Flash (a reading of MiniMax's launch table image), and Muse Glimmer (Qubrid). Their
records say so.

Leaving out each of 190 results from models with several results, then refitting and
predicting it, produced:

| Absolute prediction error | Percentage points |
| --- | ---: |
| Mean | 7.0 |
| Median | 4.9 |
| 90th percentile | 16.6 |
| Largest | 27.9 |

These are errors predicting held-out benchmark results, not confidence intervals for
the final scores. Several withheld results belong to the same model, so they are not
independent. Models with a single available benchmark cannot be validated this way.
This check measures internal consistency, not performance in SlipAgent.

Harnesses, reasoning budgets, quantization, and benchmark dates vary between
sources. LiveCodeBench releases are pooled; Terminal-Bench 2.0 and 2.1 have separate
offsets. Models whose evidence is a single old benchmark, specialist models, and
single-benchmark extrapolations carry greater uncertainty. Scores separated by a few
points should be treated as close estimates. Where benchmarks disagree about a model
the fit settles between them. gpt-oss-120b and 20b score 52 and 53 rather than their SWE
Verified 62.4 and 60.7, because their Terminal-Bench 2.0 and LiveCodeBench results
are much lower than those SWE Verified results imply.

No score is carried over from an earlier review: every score in the table has recorded
results, and the update command refuses a scored entry that has none. A model with no usable
coding result is a reviewed gap with no score, which keeps it manually selectable and out of
automatic rotation: Nemotron 3.5 Bolt E4B, Llama 3.1 Nemotron 51B, Glyph Cluster, SEA-LION,
and gpt-oss-safeguard. Searches found no coding result for the first three. Bolt's card
compares it with Gemma 4 E4B-it, Qwen3.5 4B, and Nemotron 3 Nano 4B but reports no coding
benchmark, the 51B's card lists only MT-Bench and general benchmarks, and Glyph Cluster has an
aggregator intelligence score with no methodology.

Dedicated vision models are in the model blacklist, not scored: Llama 3.2 11B/90B Vision,
Phi-3 Vision, and Nemotron 3 Nano Omni report no coding results of their own. Nemotron 3.5
Super VL stays scored because NVIDIA describes it as a model "for reasoning, coding, tool use,
document understanding, and video understanding" and reports SWE-bench, LiveCodeBench, and
Terminal-Bench results for it.

Leanstral 1.5 is scored from one FLTEval result (28.9), calibrated on Claude Haiku 4.5, Sonnet
4.6, Opus 4.6, and Kimi K2.5, which report FLTEval in Mistral's
[Leanstral post](https://mistral.ai/news/leanstral). Refitting from that post reproduces the
previous 79.5–79.7. It is a Lean proof specialist, and a proof result may overestimate broad
software ability; the score is a proof-engineering placement, not evidence of general coding
strength. Gemini 3 Flash combines Google's 78% SWE Verified with Terminal-Bench 2.0 (47.6%)
and 2.1 (58.0%), and scores 76.1.

SEA-LION v1 7B Instruct remains the one unscored model. Its published linguistic
evaluations and code-security risk measurements do not establish a comparable
code-correctness result. Its `None` score keeps manual selection available and
prevents an unsupported automatic placement.

## MT-Bench Coding Conversion

Use the **coding category averaged across both turns**, on its native 1–10
rating scale. Overall MT-Bench averages mix unrelated categories. This is a
judge rating, not an execution pass percentage.

The measurements file pairs the rating with existing results for six models: Llama 2 70B Chat, Mistral 7B Instruct v0.3, Llama 3.1 8B Instruct,
Gemma 7B IT, and Llama 3.2 1B/3B Instruct. Their HumanEval and BigCodeBench
results sit next to their rating, and the rating is fitted
as one more benchmark with a 1–10 scale (declared under `scales`). Its offset is
**−2.30**. The rating calibration covers 1.0–6.3; higher ratings require extrapolation.
Five ratings come from one report and three models are related Llamas, so the sample
provides limited independent evidence.

[Zyphra's Figure 9](https://arxiv.org/html/2411.15242v1#S6.F9) supplies approximate
chart readings for Zamba2. Its roughly **5.2/10** gives **9.8/100**. Reading the chart
as 5.1–5.3 changes that by about half a point. The sixth rating is calculated from
LMSYS's 20 published coding judgments for Llama 2 70B Chat. No new inference is run.

## Add measurements and recalculate

The [measurements file](../scripts/model-score-measurements.json) is the source of
truth. Each record names a model, its published results by benchmark, and HTTPS
source links. Results use the 0–100 scale unless the file declares another range
under `scales`. Mark a model `"reference": true` to use it for calibration without
adding it to the score table. The saved [calibration](../scripts/model-score-calibration.json)
holds the fitted offsets, counts, and validation statistics. It is maintenance data;
normal SlipAgent requests never load it. The
[update command](../scripts/update-model-scores.py) needs only standard Python and
performs no network requests or inference.

1. Verify the exact model/version and published text-coding result. Match the
   benchmark name, version, task, and metric. Use pass@1 rather than pass@k or
   best-of-many results. Keep the source link and note which results are
   developer-reported or third-party.
2. Add or edit records in the measurements file. For a benchmark no other model has
   yet, add reference models that report it together with a calibrated benchmark;
   they need not be available for free.
3. Preview, then apply:

   ```bash
   .venv/bin/python scripts/update-model-scores.py --dry-run
   .venv/bin/python scripts/update-model-scores.py
   ```

   The command prints each model's previous and new score, the offsets, and the
   leave-one-out error, then rewrites the score table and the calibration together.
   It validates the complete input first. Empty evidence, benchmarks it cannot connect
   to SWE Verified, nonfinite or out-of-range values, repeated model IDs, and
   conflicting aliases leave both files unchanged. List calibrated benchmark names
   with `--list-benchmarks`.

A refit can move existing scores, because every model and offset are fitted together.
A scored entry with no record in the measurements file is an error; reviewed gaps with no score are allowed. Provider
aliases share their model's value, while endpoint pricing still controls eligibility.

Run the focused score/routing tests and the repository's normal checks, regenerate
the hash manifest, and commit the table with any new aliases. A scored model
enters automatic rotation when a configured provider advertises an eligible free
endpoint. Removed endpoints keep their scores for a later return. New scores
take effect on catalog access after a compatible reload; use `/model reset`
to select the current highest-scoring model immediately.
