# Agent search comparison lab

Compare **AFlow**, **ADAS Meta Agent Search**, and a single-model baseline across math, reading, coding, and tool-use tasks. This project implements the five-step protocol:

1. Evaluate a fixed baseline.
2. Give both searches identical public examples, validation tasks, model access and tool capabilities.
3. Search within the same limits on dollars, tokens, calls and time.
4. Freeze the best fully evaluated validation candidate; evaluate it on held-out test tasks.
5. Repeat using the same list of seeds and report each run plus mean and sample standard deviation.

The initial smoke report uses synthetic tasks and deterministic model replies. It verifies execution and reporting, not which research method is better. The Codex backend has subsequently been checked with real ChatGPT-authenticated model requests; those connection checks are not benchmark comparisons. No API-key model calls were used for these checks.

Offline reports show software-check status only, with no method leaderboard or learning curves. The fixture deliberately returns correct answers, wrong answers and failures to verify scoring. Its raw scores are controlled test outputs, not agent performance. Earlier all-correct samples produced perfect raw scores by construction.

## Run the offline check

From this directory:

```sh
.venv/bin/python -m searchlab run --config configs/smoke.json
```

The Python environment is already installed in this workspace. For a fresh checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e . -e vendor/agentdojo
.venv/bin/python -m searchlab verify-sources
.venv/bin/python -m unittest discover -s tests -v
```

The offline check covers three task families × three methods × three seeds. It exercises the **actual upstream search code** with fake model replies. The local worker is used only for deterministic/trusted software checks. Live generated code always runs inside Docker.

## Prepared benchmarks

The local `data/` directory already contains four reproducible pilot datasets, each with **4 public training examples, 12 validation tasks, and 20 held-out test tasks**. The split seed is 1729 and is independent of the search seeds 11, 22 and 33.

| Dataset | Source / partitions | Metric |
|---|---|---|
| GSM8K | Official training data supplies public/validation examples; official test data supplies final tests | Numeric final-answer accuracy |
| DROP | Bundled ADAS development data, split by passage; unused questions from a reserved passage are discarded | Upstream DROP F1 / 100, with exact match also logged |
| MBPP | Task IDs 601–974 for public examples; 511–600 for validation; 11–510 for test | One submitted program per task, all provided tests must pass |
| AgentDojo workspace | User-task IDs split once, with all attack variants of each user task kept together | Clean utility, attacked utility and attack success |

The small pilot sets are for establishing the experiment. Increase them before drawing research conclusions. DROP uses a development-data holdout, not the unavailable official hidden test set. The MBPP check is task-level test success, not an unbiased multi-sample pass@k estimate.

To recreate or enlarge datasets:

```sh
.venv/bin/python -m searchlab prepare gsm8k --output data/gsm8k.json
.venv/bin/python -m searchlab prepare drop --output data/drop.json
.venv/bin/python -m searchlab prepare mbpp --output data/mbpp.json
.venv/bin/python -m searchlab prepare agentdojo_workspace --output data/agentdojo_workspace.json
```

Add `--train-size`, `--validation-size`, `--test-size` and `--split-seed` as needed. GSM8K and MBPP downloads are cached under `data/raw/`. You can use `--train PATH` and, for GSM8K, `--test PATH` instead. AgentDojo banking, travel and Slack are also supported by `prepare`; add their output files to your run configuration. Dataset fingerprints and split/group checks prevent silent edits and overlap between partitions.

## Configure a real experiment

### Use your existing Codex login

For the prepared all-benchmark short run, use `configs/codex-all-short.json`. It adds a shared `budgets.global` ledger, per-benchmark budget slices, one validation and one held-out user task per benchmark, and skips redundant validation of the predefined baseline. All methods and phases count toward the global 500,000-token threshold. Live cumulative accounting is in `total_budget.json`; finalized totals are also in the manifest. Codex's final usage is known only after a turn; the gateway leaves a conservative request reserve before launching near the global limit. `New candidates` in reports distinguishes validated generated designs from initial archive agents.

The **Codex CLI backend is implemented**, including structured responses and benchmark-tool relay. It uses your saved ChatGPT login, with no API key. See [the short Russian guide](START_CODEX.ru.md).

This workspace already has `configs/codex.json`, using the model selected in your local Codex settings. To create a configuration on another machine:

```sh
.venv/bin/python -m searchlab init-codex
.venv/bin/python -m searchlab check-codex --config configs/codex.json
.venv/bin/python -m searchlab plan --config configs/codex.json
.venv/bin/python -m searchlab run --config configs/codex.json --live
```

`init-codex` creates a one-seed, one-iteration pilot over real GSM8K data without calling a model. `check-codex` makes three real requests. Use `init-codex --benchmarks gsm8k drop mbpp agentdojo_workspace --output configs/codex-multi.json` for a pilot across task families. Add `--model NAME` to choose a model explicitly. Existing configurations are never overwritten by initialization.

The Codex backend sets `max_usd` and USD rates to `null`: subscription cost is **unknown**, never zero. Budgets use invocation count, tokens and time. Token caps are checked between invocations; Codex's hidden prompt overhead means a call can cross the cap, in which case the current evaluation stops. `calls` counts fresh CLI launches, not underlying HTTP requests; CLI networking can retry internally. Missing usage is flagged incomplete. Complete raw usage events, including any additional provider fields, are retained in `codex_events/`. The gateway does not retry failed invocations.

Each invocation uses an empty temporary working directory, fresh session, saved ChatGPT authentication, and explicit tool restrictions. User configuration, project instructions, native shell/apps/plugins/browser tools and web search are disabled. Benchmark tools travel through a JSON response envelope and are executed by the shared evaluator. Unexpected native tool events invalidate the invocation. Codex's model wrappers and system prompt are part of this backend, so comparisons with the API backend require separate runs. Generation seed and temperature are not controlled through the CLI; both methods use identical Codex settings.

### Use a model API

AgentDojo supplies simulated tools, tasks, attacks, scoring and model connectors. It does **not** supply a free running model. The API backend uses one shared OpenAI-compatible gateway for both research methods so that all requests are metered consistently. You can point it at a model API or a compatible local model server.

1. Copy `configs/live.example.json` to `configs/live.json`.
2. Set the search model and execution model. The two models may differ, but each role is identical for both methods. Use the same model in both roles if desired.
3. Set your provider's input/output prices per million tokens. Set zero only for genuinely free/local execution. Keys stay in the named environment variables, never in the JSON file.
4. Choose the budgets, datasets, seeds and iteration ceiling. API model support varies: configure `send_temperature`, `send_seed`, and `output_limit_parameter` for your endpoint. Inference seeds are best effort and do not guarantee deterministic remote models.
5. Start Docker and validate the executor:

```sh
.venv/bin/python -m searchlab check-executor --pull
```

The runtime image has already been downloaded and the executor checked in this workspace. Real runs resolve the image to its immutable local image ID and record it in the manifest.

Set the key in your terminal using your provider's normal setup. Review the experiment before starting it:

```sh
.venv/bin/python -m searchlab plan --config configs/live.json
.venv/bin/python -m searchlab run --config configs/live.json --live
```

The API template has $5 **per search method per benchmark per seed**, and $2 per evaluation phase. Across four benchmarks and three seeds, its maximum configured total is **$216**: two searches plus baseline validation and three final evaluations for each benchmark/seed. These are example ceilings, not an approved spend or an expected cost. Reduce them for your first live run. API experiments require a configured model, rates, key and explicit `--live`; Codex experiments use saved ChatGPT login instead.

## What is preserved from AFlow and ADAS

The repository snapshots in `vendor/` are unmodified. `vendor/manifest.json` records their source revisions, archive checksums, and source-file checksums.

- **AFlow:** load and execute the upstream `Optimizer._optimize_graph`, `DataUtils`, `ExperienceUtils`, `GraphUtils` and `EvaluationUtils`. This retains its score-weighted parent selection, workflow mutations, duplicate-modification checks and tree experience. The shared runner replaces the outer sleep/retry loop with a common finite iteration/budget policy. Failed search steps are logged and count against the iteration ceiling.
- **ADAS:** load and execute the upstream `_drop/search.py::search`. This retains all seven initial agent designs, the growing archive, both reflection steps, evaluation/debugging loop and archive fitness feedback. The DROP-specific introduction is replaced with the current benchmark's description and public examples. Its upstream percentage formatter receives normalized scores multiplied by 100.
- **Both:** replace model clients, domain data loading and candidate execution with the same gateway/evaluator. Use a common restricted Python interface, with LLM calls, basic control flow and tools. AFlow's Custom/AnswerGenerate/ScEnsemble and ADAS's Info/LLMAgentBase are compatibility adapters over that gateway. Their prompts and output formatting differ, and those differences are part of the adapted comparison. Models cannot switch themselves to a different provider or executor.

This is an **adapted comparison of upstream search methods**, not a claim to reproduce the paper's original results. The shared runtime excludes arbitrary imports in generated agent orchestration, unrestricted Python execution, the AFlow Programmer/Test operators and direct API access. Code submissions may use the allowlisted Python standard-library modules in `searchlab/worker.py`; other dependencies score as execution failures. Both searches are told these constraints.

ADAS's initial archive costs more than AFlow's single initial workflow. Those calls, candidate evaluations, reflections and debugging calls all count against the same search budget. The iteration ceiling can stop either method before it spends the entire budget. Learning curves therefore show performance against actual search model calls, not just generation number. Compare runs at matched budgets, not only matched iteration counts.

## Isolation and measurement

Live generated agents and code tests execute in containers with no network, a read-only filesystem, no project or dataset mounts, no API credentials, an unprivileged user, dropped capabilities, CPU/memory/process limits and task timeouts. The only mounted file is the worker runtime. A parent process relays allowed model/tool requests; it holds hidden answers and AgentDojo environment state. AST checks define the shared interface; Docker is the isolation boundary.

Every task starts with fresh state. AgentDojo tools use the benchmark's simulated environment, not real email or banking services. Multiple modules within one task share state; branching into alternative workflows does not clone or roll back side effects. Tool calls and environment changes remain visible to the official scoring checks.

For optional AgentDojo attacks, each user task gets a clean evaluation and one evaluation per configured injection goal. Only fixed built-in attack templates are accepted, so each candidate sees the same attack construction. The default search objective averages clean task utility and attacked **robust utility** (`user task succeeds AND attack fails`). Raw clean utility, attacked utility and attack success remain separate in the results. Lower attack success is better; the upstream field named `security` means injection success during attacked runs. Change `agentdojo_objective` to `utility` to optimize only task success.

No final-test examples, labels or scores are given to the optimizers. Artifacts for all methods within a benchmark/seed are saved before any of their final tests run. Selection uses only completed validation evaluations. Interrupted final tests have **no aggregate score**, rather than a misleading average over the completed prefix.

Budgets are checked before each provider request using conservative input-byte and maximum-output reservations. Calls are serialized. Providers that omit usage or return ambiguous errors stop that phase, with the reservation charged and cost marked uncertain; there are no invisible SDK retries. Costs use configured uncached rates and are estimates, not invoices. Pricing errors, extra provider charges, or providers that exceed the declared token envelope cannot be prevented by a client-side dollar meter. Time limits are enforced before requests and through provider/task timeouts; an in-flight request can finish after the phase deadline. Use provider-side account limits as the final billing ceiling.

## Results

Every run gets a new directory; existing runs are never overwritten:

```text
runs/<run-id>/
  manifest.json             configuration, fingerprints, packages and image ID
  results.json / results.csv
  report.md / report.html   summary and per-seed validation learning curves
  <benchmark>/splits.json
  <benchmark>/seed_<seed>/<method>/
    history.json           fully evaluated validation candidates
    incumbent.json         best candidate so far
    candidates/            source for every fully evaluated candidate
    frozen.json            selected source, checksum and validation provenance
    search_calls.jsonl     proposal and candidate-execution usage
    test_calls.jsonl       held-out inference usage
    codex_events/          raw events for each Codex invocation, when selected
    tasks.jsonl            scores, outputs, errors and tool traces
    summary.json
    upstream/              original search archive/tree artifacts
```

The HTML report is self-contained and makes no external requests. Regenerate it with:

```sh
.venv/bin/python -m searchlab report runs/<run-id>
```

Reports show per-benchmark scores, paired gains over baseline, seed variability, cost, errors and completion status. They do not average incompatible task metrics into one overall ranking. Search and final-agent execution costs are separate. Review incomplete/error/uncertain-cost runs before making comparisons. To change configurations or repeat a run, create a new run; resumable live searches and distributed execution are not implemented.

## Sources and licensing

- [AFlow code](https://github.com/FoundationAgents/AFlow), [paper](https://arxiv.org/abs/2410.10762): MIT, retained in `vendor/aflow/LICENSE`.
- [ADAS code](https://github.com/ShengranHu/ADAS), [paper](https://arxiv.org/abs/2408.08435): Apache 2.0, retained in `vendor/adas/LICENSE`.
- [AgentDojo](https://github.com/ethz-spylab/agentdojo): MIT, retained in `vendor/agentdojo/LICENSE`.
- [GSM8K data](https://github.com/openai/grade-school-math), [MBPP data](https://github.com/google-research/google-research/tree/master/mbpp), and the upstream ADAS DROP metric/data retain their original provenance and terms.
- The shared model gateway uses the [official Chat Completions API schema](https://platform.openai.com/docs/api-reference/chat/create), with configurable endpoint compatibility.

Synced project reference files and the older `autoresearch/` and `tuning_lab/` projects are untouched.
