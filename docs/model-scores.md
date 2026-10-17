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

Four reviewed models have no verified text-coding evidence: SEA-LION 7B
Instruct, Zamba2 7B Instruct, and Llama 3.2 11B/90B Vision. Their identities remain
in the table with `None` scores. They are manually selectable; visual results
cannot justify automatic text-coding placement.

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
   may contain one object or an array of objects. Percentages use the 0–100 scale.
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
