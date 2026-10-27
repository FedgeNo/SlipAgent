# Estimated coding scores

SlipAgent ranks eligible free endpoints using one synthetic text-coding score
per model, from 0 to 100, with higher values preferred. The score is an estimate
for comparing models across different coding benchmarks. It is not a measured
pass rate or a probability that SlipAgent will complete a task.

Only scores and exact model aliases enter runtime metadata. Free pricing, keys,
provider availability, context limits, and request failures remain separate
selection checks. A paid model can help calibrate scores without becoming an
automatic routing candidate. Visual benchmark results contribute nothing.

## Calibration reviewed October 26, 2026

The scores are computed from raw published results saved in
[`scripts/model-score-measurements.json`](../scripts/model-score-measurements.json):
176 results for 88 models, including 40 reference models that are not scored for
routing. Eight text-coding benchmarks are connected through shared models.
SWE-bench Verified supplies the reference scale. DeepSWE 1.1, Terminal-Bench 2.0
and 2.1, LiveCodeBench, HumanEval, BigCodeBench complete, and the MT-Bench coding
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
| SWE Verified | 0.00 | 26 | 26 |
| DeepSWE 1.1 | +2.21 | 26 | 25 |
| Terminal 2.0 | +1.13 | 11 | 11 |
| Terminal 2.1 | +0.76 | 37 | 34 |
| LiveCodeBench | −0.67 | 15 | 14 |
| BigCodeBench complete | −0.78 | 30 | 23 |
| HumanEval | −2.04 | 24 | 20 |
| MT-Bench coding | −1.77 | 7 | 6 |

The offsets agree with the previous review, which used a different, partly
unrecorded set of results: every offset moved by under 0.3 log-odds, and the
scores of the new-generation models moved by one to two points.

Primary sources are the models' own cards and technical reports, the
[EvalPlus](https://evalplus.github.io/results.json) and
[BigCodeBench](https://bigcode-bench.github.io/results.json) result files, and
Google's, Moonshot's, Poolside's, and Alibaba's comparison tables for reference
models. Every record keeps its source links. Several 2026 results are developer-reported
and not independently reproduced. Four are third-party quotations rather than a
developer page: Step 5 Preview (BenchLM), dots-3 note preview (BenchLM), MiniMax M3.1
Flash (a reading of MiniMax's launch table image), and Muse Glimmer (Qubrid). Their
records say so.

Leaving out each of 159 results from models with several results, then refitting and
predicting it, produced:

| Absolute prediction error | Percentage points |
| --- | ---: |
| Mean | 6.5 |
| Median | 4.5 |
| 90th percentile | 16.8 |
| Largest | 28.3 |

These are errors predicting held-out benchmark results, not confidence intervals for
the final scores. Several withheld results belong to the same model, so they are not
independent. Models with a single available benchmark cannot be validated this way.
This check measures internal consistency, not performance in SlipAgent.

Harnesses, reasoning budgets, quantization, and benchmark dates vary between
sources. LiveCodeBench releases are pooled; Terminal-Bench 2.0 and 2.1 have separate
offsets. Models whose evidence is a single old benchmark, specialist models, and
single-benchmark extrapolations carry greater uncertainty. Scores separated by a few
points should be treated as close estimates. Where benchmarks disagree about a model
the fit settles between them. gpt-oss-120b and 20b score 52 rather than their SWE
Verified 62.4 and 60.7, because their Terminal-Bench 2.0 and LiveCodeBench results
are much lower than those SWE Verified results imply.

Twenty-seven catalog entries have no results in the measurements file, so the fit
does not touch them. They keep their previously reviewed scores, which came from the
earlier calibration: `mistralai/mistral-large-2-instruct`, the Gemini 2.5 Flash and
Flash-Lite and Gemini 3 Flash entries, Leanstral, Nemotron 3.5 Super VL, Lightning
and Bolt, and the older Nemotron, Phi, Jamba, Gemma 3, Granite, and CodeGemma
entries, among others. The FLTEval and description-only benchmarks from the earlier
review are no longer calibrated because their per-model results were not recorded
with it. Leanstral's score is carried over unchanged for that reason.

Nemotron 3.5 Bolt E4B has no published coding result, so its 34.7 is a peer-derived
placement, not a measurement. Its [model card](https://huggingface.co/nvidia/Nemotron-3.5-Bolt-E4B)
measures it beside Gemma 4 E4B-it, Qwen3.5 4B, and Nemotron 3 Nano 4B under one
harness, and those peers report LiveCodeBench 52.0, 55.8, and 51.8. On the card's
tool-use rows Bolt falls between Gemma and Qwen each time, which places it near 54.2 on
the LiveCodeBench scale. The peer range bounds it to roughly 32.5–36.2. Replace it when
NVIDIA or an independent evaluation publishes a coding result.

SEA-LION v1 7B Instruct remains the one unscored model. Its published linguistic
evaluations and code-security risk measurements do not establish a comparable
code-correctness result. Its `None` score keeps manual selection available and
prevents an unsupported automatic placement.

## MT-Bench Coding Conversion

Use the **coding category averaged across both turns**, on its native 1–10
rating scale. Overall MT-Bench averages mix unrelated categories. This is a
judge rating, not an execution pass percentage.

The [saved paired evidence](../scripts/model-score-mt-bench.json) contains six
models: Llama 2 70B Chat, Mistral 7B Instruct v0.3, Llama 3.1 8B Instruct,
Gemma 7B IT, and Llama 3.2 1B/3B Instruct. Their HumanEval and BigCodeBench
results are in the measurements file next to their rating, and the rating is fitted
as one more benchmark with a 1–10 scale (declared under `scales`). Its offset is
**−1.77**. The rating calibration covers 1.0–6.3; higher ratings require extrapolation.
Five ratings come from one report and three models are related Llamas, so the sample
provides limited independent evidence.

[Zyphra's Figure 9](https://arxiv.org/html/2411.15242v1#S6.F9) supplies approximate
chart readings for Zamba2. Its roughly **5.2/10** gives **15.6/100**. Reading the chart
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
   .venv/bin/python scripts/update-model-scores.py --fit --dry-run
   .venv/bin/python scripts/update-model-scores.py --fit
   ```

   The command prints each model's previous and new score, the offsets, and the
   leave-one-out error, then rewrites the score table and the calibration together.
   It validates the complete input first. Empty evidence, benchmarks it cannot connect
   to SWE Verified, nonfinite or out-of-range values, repeated model IDs, and
   conflicting aliases leave both files unchanged. List calibrated benchmark names
   with `--list-benchmarks`.

A refit can move existing scores, because every model and offset are fitted together.
Entries with no record in the measurements file keep their reviewed score. Provider
aliases share their model's value, while endpoint pricing still controls eligibility.

To add one model quickly without refitting everything, the original single-record
form still works: copy [the example input](../scripts/model-score-example.json), then
run `update-model-scores.py new-model.json --dry-run`. It holds the saved offsets fixed
and changes only that model, and it rejects a benchmark that the calibration does not
contain. Put the same record in the measurements file at the next refit so the model
keeps its evidence.

Run the focused score/routing tests and the repository's normal checks, regenerate
the hash manifest, and commit the table with any new aliases. A scored model
enters automatic rotation when a configured provider advertises an eligible free
endpoint. Removed endpoints keep their scores for a later return. New scores
take effect on catalog access after a compatible reload; use `/optimal reset`
to select the current highest-scoring model immediately.
