# Free Models, Optimal Mode, and Providers

[Back to the README](../README.md)

## Using Free Models

**Optimal mode is enabled by default.** It selects from known, ranked free
coding models with configured API keys. General-purpose routers such
as `openrouter/free` and `kilo-auto/free` remain manual choices because their
underlying model changes.
A large context window gives the agent room for project instructions, source,
tool results, and a long-running conversation as it develops a project or its
own harness.

| Example Free Model | OpenRouter Model ID |
| --- | --- |
| [NVIDIA Nemotron 3 Ultra (free)](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free) | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| [NVIDIA Nemotron 3 Super (free)](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) | `nvidia/nemotron-3-super-120b-a12b:free` |

### Optimal Mode

Ordinary startup selects the highest-ranked eligible free model. Explicit
`--model` or `--provider` launch options start in manual mode.

Interactive startup does not wait for the network. Provider catalogs and the
selected models' capability profiles are saved in `startup-cache.json` in the
data directory, for each API key and endpoint, for up to seven days. When they
are available, startup selects from them and refreshes them in the background,
along with the quota. A model that has since become unavailable fails its first
request (for example with 403 or 404) and optimal mode moves straight to the
next model. The first interactive launch without any API key asks for one,
lists providers with free models, and offers to save it. `/optimal on`
enables automatic selection again. Model metadata stores one **estimated text-coding
score from 0 to 100; higher is better**. Published benchmark results are calibrated
onto a common scale using models tested on multiple benchmarks, including paid
reference models. Visual results are excluded. These synthetic scores guide
selection; they are not measured pass percentages or guarantees of results.
The same model/version shares its score across providers, and model lists display
it. The complete table is in [`src/slipagent/modelscores.py`](../src/slipagent/modelscores.py).
See [scoring and updates](model-scores.md) for sources, validation limits, and
the offline tool that scores a new model using saved calibration without
recalculating existing entries.
Provider-specific IDs, including Step 5's free promotional route and NVIDIA's
longer Nemotron IDs, share the underlying model's score. A score does not make a
paid endpoint free: automatic selection still checks that endpoint's pricing.

Working requests keep their ordinary retries for transient failures. An
unusable model, permanent request failure, or exhausted recovery moves coding
to the next highest-scoring eligible model. The conversation, original task,
failed request records, and completed tool results stay intact. Tool/test failures remain
feedback for the current model. Each automatic selection is announced.
Connection failures retain the current model after its ordinary retries are
exhausted. A broken network connection does not exclude a model or provider;
retry when connectivity recovers. HTTP errors such as 429 and 503 still count
as provider failures after retry exhaustion.
After 10 seconds without model output, the status line counts down to the
deadline; in optimal mode, **Ctrl+N** skips to the next model immediately.
A working request that produces no output within two minutes is cancelled and
moves directly to the next eligible model. This deadline runs from request start,
including connection and upload time but not upload pacing waits; keepalives and empty stream markers do not
extend it. Once reasoning, answer text, or tool-call content arrives, streaming
continues under the ordinary transport timeout. In manual mode, the deadline
reports an error and retains the selected model. Background compression retains
its separate 20-minute timeout.

The replacement remains selected until another failure, manual selection, or
reset. Both `/optimal on` and `/optimal reset` clear failed coding/compression
routes and retry from the highest score. `/optimal next` excludes the current
model until reset and selects the next ranked one; with optimal off, it starts
optimal like `/optimal on`. `/optimal off` keeps the current model
under manual control; `/optimal` reports the state. Successful manual model or
provider selection through `/model`, `/models`, or `/provider` also disables
optimal mode. During active work, `/optimal` changes apply at the next safe step
boundary in input order, without stopping the task. Opening or cancelling a
picker leaves the mode unchanged. Optimal
mode is session-only; ordinary saved model preferences remain unchanged by
automatic selection and remain available when launching with `--provider`.

When every eligible model fails, coding stops with an explanation. Another
prompt cannot reuse an excluded model: use `/optimal reset`, update credentials,
or select a model manually. Failure exclusions are session-local and clear on
a new conversation; a credential change clears that provider's exclusions.
An authentication failure excludes that provider for both coding and compression;
model-specific failures are tracked separately for those two uses.

New free catalog models without a score remain manually selectable. SlipAgent
announces their release and asks you to update SlipAgent for updated scores.
Scores are refreshed with software updates, without model calls or automatic
benchmark collection at runtime. The October 16 review scores 68 models, including
Reflection Beam, Xing 4.0, Zamba2, and the Llama 3.2 11B/90B Vision models.
Zamba2 uses a calibrated MT-Bench coding rating; the Llamas use their preserved
text-model results. SEA-LION v1 7B Instruct remains unscored because no comparable
code-correctness result was found. Known evidence gaps do not trigger new-release notices.
With `GEMINI_BILLING_TIER=paid`, Google's billed Gemini IDs are excluded from automatic
free selection; the always-free Gemma 4 models remain eligible.

Provider context limits and free availability can change. Check `/models` for
live catalog information; selection also fetches endpoint properties and uses
their more conservative limits. A model's advertised maximum does not guarantee
that every free endpoint offers the same window.
Automatic routing refreshes provider catalogs after five minutes. Removed models
keep their stored scores but are excluded from coding and compression until
they return. A failed catalog connection retains that provider's last known
inventory; a successful empty catalog removes its models from selection.

Model listings hide a curated blacklist of dedicated non-coding models, maintained
with reasons in `src/slipagent/model_catalog.py`. New model IDs remain visible by
default; missing tool-call or structured-output metadata does not exclude them.
This applies to `/models`, the `/model` picker, and `--list-models`.
Searches match every whitespace-separated word across the model ID, name, and
provider, ignoring case and word order (for example, `/models ultra nemotron`).

Use `/model free` to select from zero-cost catalog entries. Use Up/Down and
Enter to select, or Escape to cancel. `/models <text>` searches model IDs, names, and providers across active provider catalogs. Results include the provider; a provider is active only when its API key is configured. `/model` opens the full catalog; `/model <provider>::<model>` selects a specific provider and model. A bare model ID also works when it is unambiguous. Successful selections are remembered for future launches. Paid OpenRouter models also work.
Without an interactive terminal, model searches and partial-name suggestions use ASCII tables formatted with
`tabulate`, showing model IDs, context limits, and input/output prices. Long IDs
wrap inside their cells; narrow terminals display each model's fields vertically.

SlipAgent is designed around free OpenRouter inference. Model requests are a
limited resource, so the prompt encourages batching predictable tool calls into
one response. Up to four independent file/search reads run concurrently. Writes,
shell commands, state updates, and unclassified MCP tools form ordered barriers;
all observations return in the model's declared order. A failed
tool contributes an error result instead of discarding the rest of the batch.
The model can explain its work alongside those calls.

Free inference still has availability and request limits; see
[OpenRouter's current limits](https://openrouter.ai/docs/api_reference/limits).
SlipAgent refreshes key information at startup and every 15 minutes, including
while idle or working. The footer shows the free-call count when supplied by
OpenRouter; `/key show` displays key information and `/cost` shows session token
usage and reported spend.

## API Providers

OpenRouter remains the default. For NVIDIA, add `NVIDIA_API_KEY=your-key` to the ignored `.env`, then run:

```bash
slipagent --provider nvidia
```

NVIDIA defaults to `nvidia/nemotron-3-ultra-550b-a55b`, without OpenRouter's `:free` suffix. No local GPU is needed. Inside an existing session:

```text
/models nemotron
/model nvidia::nvidia/nemotron-3-ultra-550b-a55b
```

Search reads only providers with configured API keys, including keys supplied for the current session. Providers without keys are neither queried nor listed. Duplicate model IDs remain separate choices. A provider's catalog failure is reported while results from the others remain available. Selection checks credentials are present, loads capabilities, and validates the context budget before replacing the current client. Same-provider changes let existing summaries continue; provider changes cancel and drain pending summaries before closing their client. Originals remain available. Authentication is established by an authenticated API request, not by catalog access.

For Kilo Gateway, add `KILO_API_KEY=your-key` to the ignored `.env`, then run
`slipagent --provider kilo`. Its default is `kilo-auto/free`, Kilo's automatic
free-model selector. `/model free` searches the active catalogs;
`/model kilo::<model-id>` selects an exact Kilo model. SlipAgent fetches Kilo's
`/models` endpoint for model IDs, prices, context/output limits, tool support,
and reasoning options. Advertised reasoning efforts use the highest supported
level; otherwise reasoning is enabled when advertised. Catalog entries and free
availability can change. Missing models are rejected without substituting a paid
model. Kilo does not supply account quotas through this integration.

For Mistral, add `MISTRAL_API_KEY=your-key` to your local `.env`, then run
`slipagent --provider mistral`. The default is `mistral-small-latest`;
`/models mistral` lists available chat models and
`/model mistral::<model-id>` selects one. Mistral's authenticated catalog supplies
context limits, aliases, and native tool support. Leanstral 1.5 and 1.5.1 are labelled free
and share the same estimated coding score, including when loaded from the startup cache;
other prices remain unknown because the catalog does not identify your account's free allowance. Free-mode access
and quotas are controlled by Mistral. Its ranked free models participate in
background compression. `/provider` switches only to defaults advertised
as free, so select Mistral models explicitly with `/model`, for example
`/model mistral::labs-leanstral-1-5-1` when available in your account's catalog.

For Google Gemini, add `GEMINI_API_KEY=your-key` to your local `.env`, then run
`slipagent --provider google`. The default is `gemini-3.8-flash`.
`/models google` lists compatible text models with Google's input and output
limits. Tool calls use SlipAgent's embedded JSON protocol so compressed history
does not depend on Google's opaque native function-call signatures.
Google's free Gemini tier and always-free Gemma 4 models participate in
background compression when eligible.
SlipAgent requests thought summaries from Google's thinking models and displays
them in the reasoning stream, using the model's default thinking level.
Leading `<thought>…</thought>` blocks are separated from response content before
tool-call JSON parsing, including when their tags span streaming chunks.
Google free-tier model uploads are paced at 200,000 estimated input tokens per
minute, including retried uploads. Paid-only models upload at full speed.
Set `GEMINI_BILLING_TIER=paid` for a billed Google account to upload billed models
at full speed; always-free Gemma requests retain pacing. The default is `free`, since the catalog does not report account
billing status. Concurrent paced uploads through the same client share the
allowance. Responses continue streaming at the provider's speed. Paced models
limit each prompt to the 200,000-token allowance, and pacing waits do not count
toward the first-output deadline.

Model lists put free models first, ordered by descending estimated coding score,
with unscored entries afterward. Ties use provider and model ID for stable ordering.
The interactive picker uses a green background
and a matching `= free` legend on the right. `/model free` includes models with
a free tier; these retain a `free tier` label because account billing and quotas
still apply. Unknown prices are shown as `-`.
Search matches every word against the displayed model row; for example,
`/models google free` shows Google's free-tier models.

NVIDIA's hosted trial catalog is free, including models developed by other
companies. Its catalog omits tool support and context limits, so
[`nvidiaprofiles.py`](../src/slipagent/nvidiaprofiles.py) supplies reviewed endpoint
profiles for Step 5, DeepSeek V4.1 Flash, GLM 5.3/Flash, Kimi K2.6, current
Nemotron models, Muse Glimmer, Gemma 4, DiffusionGemma, Laguna XS 2.1, and
GPT-OSS 20B. Native tools are used where the endpoint documents them; other
profiles use embedded tool calls. Gemma endpoints that accept only user and
assistant roles receive the system instructions at the start of the first user
message, streamed from the same source files. Remaining NVIDIA entries require
an explicit capability profile before use; automatic selection skips those
whose capabilities cannot be established. Free availability and quotas remain
provider-controlled.

Python callers select the same modules programmatically:

```python
from slipagent import Config, Message, create_client

config = Config.from_env(provider="nvidia")
async with create_client(config.provider, api_key=config.api_key,
                         reasoning_effort="medium", reasoning_budget=2048) as client:
    await client.model_capabilities(config.model)
    reply = await client.chat(model=config.model,
                              messages=[Message.user("Explain this algorithm.")],
                              max_tokens=4096)
```

Pass that client to `Agent` to use the normal coding loop. `NvidiaClient(profiles={model_id: RequestProfile(...)})` supports additional explicitly verified models. Its Ultra reasoning options are `none`, `medium`, and `high` (default); `reasoning_budget` accepts -1 or 0–32,768 tokens. See [the provider contract](architecture.md#api-provider-modules) for extending the registry. Restart existing SlipAgent processes after changing the shared provider/state contracts.
