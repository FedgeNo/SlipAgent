# Estimated coding scores

SlipAgent ranks eligible free endpoints using one synthetic text-coding score
per model, from 0 to 100, with higher values preferred. The score is an estimate
for comparing models across different coding benchmarks. It is not a measured
pass rate or a probability that SlipAgent will complete a task.

Only scores and exact model aliases enter runtime metadata. Free pricing, keys,
provider availability, context limits, and request failures remain separate
selection checks. A paid model can help calibrate scores without becoming an
automatic routing candidate. Visual benchmark results contribute nothing.

## Calibration reviewed October 16, 2026

The review used 149 published measurements for 87 models, including 22 reference
models outside the scored catalog. Shared results connect nine text-coding
benchmarks. SWE-bench Verified supplies the reference scale. HumanEval,
BigCodeBench complete, DeepSWE 1.1, Terminal-Bench 2.0 and 2.1, LiveCodeBench,
FLTEval pass@1, and description-only coding supply additional evidence.

For a measured percentage `p`, convert it to log-odds:

```text
logit(p / 100) = model_ability - benchmark_offset
estimated_coding_score = 100 / (1 + exp(-model_ability))
```

Fit the model abilities and benchmark offsets jointly, fixing the SWE Verified
offset at zero. Huber loss with a 0.75 log-odds threshold reduces the influence
of conflicting results. It keeps every observation. Probabilities are clipped
to 0.005–0.995 to handle published zero/full scores. Store one decimal place in
the resulting score table. This retains useful ordering without claiming that
small differences are meaningful.

This is a bounded alternative to multiplying raw percentages by a ratio. Models
tested on both benchmarks estimate their relative difficulty. Multiple connected
benchmarks can bridge models that were never evaluated on the same test.

Primary calibration sources include the [EvalPlus results](https://evalplus.github.io/results.json)
and [BigCodeBench results](https://bigcode-bench.github.io/results.json), matched
by exact model/version identity. Published [Claude 3.5 coding results](https://www.anthropic.com/news/3-5-models-and-computer-use)
connect older code generation tests to SWE Verified. [Claude Sonnet 4.6's system card](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf)
and [Haiku 4.5 results](https://www.anthropic.com/news/claude-haiku-4-5)
connect the SWE scale to the [FLTEval comparison](https://mistral.ai/news/leanstral/).
[Leanstral 1.5](https://mistral.ai/news/leanstral-1-5/) supplies its own pass@1 result.
Mistral's live catalog lists `labs-leanstral-1-5` and `labs-leanstral-1-5-1` as
aliases; both use the resulting 79.7 estimate through the exact identity table.
The [description-only coding comparison](https://arxiv.org/html/2410.12381v1)
uses text descriptions; its image-input scores are excluded.
New catalog entries include [Reflection Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B)
and [Xing 4.0](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B).

Leaving out each of 66 known measurements for models with multiple results,
then refitting and predicting it, produced:

| Absolute prediction error | Percentage points |
| --- | ---: |
| Mean | 5.6 |
| Median | 4.3 |
| 90th percentile | 11.2 |
| Largest | 21.5 |

These are errors predicting the held-out benchmark results, not confidence
intervals for the final synthetic scores. Several withheld results belong to
the same model, so the observations are not independent. Models with a single
available benchmark cannot be validated this way. Reference-model results
remain in the fit. This check measures internal consistency, not performance
in SlipAgent.

Harnesses, reasoning budgets, quantization, and benchmark dates vary between
sources. LiveCodeBench releases/date windows are pooled in this calibration;
Terminal 2.0 and 2.1 have separate offsets. Older models, specialist proof models
such as Leanstral, and single-benchmark extrapolations carry greater uncertainty.
For example, a high proof-engineering result may overestimate broad software
engineering ability. Scores separated by a few points should be treated as
close estimates. Broadly applicable shared measurements improve future reviews.

The MT-Bench extension below adds Zamba2 at 16.2. Llama 3.2 11B/90B Vision use
the shared text-model HumanEval results, 72.6/80.5, carried into
[Amazon's comparison](https://arxiv.org/html/2506.12103v1#S2.T7), producing 28.9/38.8.
[Meta's release](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
states that the language parameters and text capabilities are preserved. These
are shared-backbone estimates, not independent coding measurements of the vision
checkpoints. Visual benchmark scores remain excluded.

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
Gemma 7B IT, and Llama 3.2 1B/3B Instruct. Their existing HumanEval and/or
BigCodeBench results determine their coding scores using the original saved
calibration. Fit the Huber location of
`logit(existing_score / 100) - logit(coding_rating / 10)`, holding the original
benchmark offsets fixed. The new offset is **−1.7248025398743945**.

Leaving each entire model out and fitting the other five produces mean absolute
error **2.75 score points**, median **2.45**, and maximum **6.33**. This check
predicts already estimated coding scores, not measured task success. Five
ratings come from one report and three models are related Llamas, so the sample
provides limited independent evidence.

[Zyphra's Figure 9](https://arxiv.org/html/2411.15242v1#S6.F9) supplies approximate
chart readings for Zamba2 and five references. Zamba2's roughly **5.2/10** gives
**16.2/100**. Reading the chart as 5.1–5.3 changes that to 15.6–16.7; omitting
one calibration model at a time changes it to 14.2–18.5. These are sensitivity
checks, not confidence intervals. The sixth rating is calculated from LMSYS's
20 published coding judgments for Llama 2 70B Chat. No new inference is run.
Calibration covers ratings 1.0–6.3; higher ratings require extrapolation.

Future inputs can use `"benchmarks": {"MT-Bench coding": 5.2}` directly. The
updater validates the 1–10 range and uses the saved conversion. Existing model
scores and benchmark offsets remain unchanged.

## Add a model without recalculating existing scores

The saved [calibration](../scripts/model-score-calibration.json) contains the
benchmark offsets, overlap counts, reviewed ranges, and validation statistics.
It is maintenance data; normal SlipAgent requests never load it. The
[update command](../scripts/update-model-scores.py) needs only standard Python
and performs no network requests or inference.

1. Verify the exact model/version and published text-coding result. Match the
   benchmark name, version, task, and metric. Use pass@1 rather than pass@k or
   best-of-many results. Keep source links in the input for the next review.
2. Copy [the example input](../scripts/model-score-example.json), replacing the
   model ID, results, source links, and optional exact provider aliases. A file
   may contain one object or an array of objects. Percentages use the 0–100 scale;
   MT-Bench coding ratings use 1–10.
3. Preview, then apply:

   ```bash
   .venv/bin/python scripts/update-model-scores.py new-model.json --dry-run
   .venv/bin/python scripts/update-model-scores.py new-model.json
   ```

   List accepted benchmark names with `--list-benchmarks`.

At least one calibrated benchmark is sufficient to calculate an estimate.
Each saved benchmark has multiple models linking it to the reference scale.
For a new model, offsets stay fixed and only that model's ability is fitted.
Existing scores remain unchanged; no list positions need renumbering. Updating
an existing canonical ID intentionally replaces that model's score. Provider
aliases share its value, while endpoint pricing still controls eligibility.

The command validates the complete input before replacing the score table. Empty
evidence, uncalibrated benchmarks, nonfinite/out-of-range values, repeated model
IDs, and conflicting aliases leave it unchanged. Source links are retained in
the input; the offline command cannot verify their claims. Keep reviewed input
files and detailed research in `agents/`.

For an unfamiliar benchmark, collect results for multiple reference models that
also have results on calibrated benchmarks, then deliberately review/refit the
calibration. The updater refuses to invent an offset. A full recalibration can
change existing scores and should be validated separately. Adding one model
does not require it.

Run the focused score/routing tests and the repository's normal checks, regenerate
the hash manifest, and commit the table with any new aliases. A scored model
enters automatic rotation when a configured provider advertises an eligible free
endpoint. Removed endpoints keep their scores for a later return. New scores
take effect on catalog access after a compatible reload; use `/optimal reset`
to select the current highest-scoring model immediately.
