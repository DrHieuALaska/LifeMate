# LifeMate

A shared Python project for a five-person team building a Vietnamese voice-to-action agent for the Zalo AI Challenge 2026. The working deliverable is an offline, **train-only evaluation benchmark**. No ASR model, LLM, fine-tuning pipeline or model-performance baseline is bundled.

## Layout

```text
.github/workflows/tests.yml   Dataset-independent Linux/Windows CI
configs/benchmark.yaml       JSON-compatible YAML 1.2 defaults
src/lifemate/
  asr/                      ASR protocol
  agent/                    Label-free request/decision protocol
  tools/                    Catalog validation, one-session runtime wrapper
  evaluation/               Dataset, trace metrics, taxonomy, strata, reports, CLI
  utils/                    Strict JSON, access guards, shared EDA heuristics
  exploration.py            Migrated reusable EDA implementation
scripts/                    Checkout-friendly entrypoints and trivial smoke generator
tests/                      Synthetic unit tests, including migrated EDA tests
docs/                       Architecture, evaluation protocol and team workflow
artifacts/                  Ignored local predictions and evaluation/EDA outputs
train+public_test/           Ignored confidential original package; never moved/deleted
```

## Setup and tests

Python 3.11+ is supported; development verification uses Python 3.14. The evaluator has no third-party runtime dependencies. Installation is optional for checkout scripts.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

Install optional plotting dependencies with `python -m pip install -e ".[eda]"` to run all EDA tests and figures. EDA tests skip cleanly without those extras. CI uses the extras and synthetic fixtures; it needs no competition dataset or GPU.

## Fixed development benchmark

Obtain the confidential package through the team's authorized channel. Keep the existing EDA manifests at `train+public_test/reports/eda/validation_split/`. The evaluator requires exactly **193 unique development IDs**, validates sibling train/holdout manifests when present, and preserves manifest order. It never creates another split or evaluates holdout records. Missing manifests produce an actionable error.

Each prediction line is an object containing `id` and `tool_calls`. Each call contains a string `name` and object `arguments`. Optional `state_hash` is ignored by trace metrics. Predictions are generated independently of the evaluator and must not receive current-request gold labels/results/states.

```powershell
python scripts/run_dev_benchmark.py --data-root "./train+public_test" --manifest "./train+public_test/reports/eda/validation_split/dev_ids.txt" --predictions "./artifacts/predictions/baseline.jsonl" --output "./artifacts/benchmarks/baseline" --mode trace
```

Choose a new output directory for every run; existing experiment outputs are not overwritten. `--limit N` evaluates a visibly labeled subset of the same manifest. `--model-id`, `--prompt-id`, `--controller-id` and `--experiment-name` record supplied provenance without inventing metadata. `--config configs/benchmark.yaml` reads the included JSON-compatible YAML subset; CLI settings override it.

For a model-free end-to-end smoke check:

```powershell
python scripts/make_empty_predictions.py --output artifacts/predictions/empty_smoke.jsonl
python scripts/run_dev_benchmark.py --predictions artifacts/predictions/empty_smoke.jsonl --output artifacts/benchmarks/empty_smoke --experiment-name trivial-empty-no-model
```

This deliberately emits no actions for every sample. Its score is **not an AI model result**.

The main metrics are exact trace accuracy, ordered tool-name accuracy, micro occurrence-level tool multiset precision/recall/F1, name edit distance, aligned argument agreement, no-action/clarification recognition, a deterministic error taxonomy and overlapping gold-derived strata. See [the precise protocol](docs/evaluation_protocol.md) before interpreting scores.

## Optional execution replay

```powershell
python scripts/run_dev_benchmark.py --predictions artifacts/predictions/baseline.jsonl --output artifacts/benchmarks/baseline_replay --mode replay
```

Replay explicitly loads the documented **train** ToolEnv once per valid sample, executes predictions only, records rejected calls, and compares evaluator-only before/after snapshots with `label.final_state`. Use `--no-state-comparison` to disable snapshots. Raw exact changed-app agreement is a diagnostic, not verified official competition scoring. `ok=true` is acceptance, not proof of task completion. Real replay accesses initial states through ToolEnv; synthetic replay tests do not read any environment files. No official-state replay was needed to verify this refactor.

## Agent integration

Implement `ASR.transcribe(Path) -> str` and/or `Agent.next_action(AgentInput) -> ToolDecision | Finish`. The agent input includes current transcript, permitted context, history, allowed tool schemas, and actual prior runtime results. It contains no current gold labels/results/final states. `build_agent_input()` makes independent copies; `EnvRunner.run_agent()` manages one environment and a bounded next-action loop without evaluator labels. Historical calls/results are legitimate history and have already been applied; never replay them. Future model adapters can emit the same prediction JSONL and reuse every metric unchanged.

## EDA preservation

The original scripts/tests/reports remain untouched under the ignored data directory. The reusable implementation and tests were copied into this project; language regexes were extracted unchanged to `lifemate.utils.language`. The migrated entrypoint accepts `README_DATA.md` and the former `README.md` filename.

```powershell
python scripts/explore_lifemate.py --root train+public_test --split train --output artifacts/eda --report artifacts/dataset_exploration.md --seed 20261029
```

This writes new local artifacts and does not overwrite the existing split manifests. Public-test EDA remains an explicit optional aggregate-only flag in the exploration tool; benchmark execution never reads public-test files. Review any newly generated split proposal before replacing the fixed benchmark manifest.

## Team and confidentiality

Member 1 owns ASR; Member 2 agent control; Member 3 model infrastructure/local LLM adapters; Member 4 tools and grounding; Member 5 evaluation/data. [Team workflow](docs/team_workflow.md), [architecture](docs/architecture.md), and [contribution rules](CONTRIBUTING.md) define shared interfaces and review expectations.

Never commit/distribute the dataset, personal information, transcripts, audio, states, prediction logs, manifests, EDA results, credentials or weights. Keep derived outputs under ignored `artifacts/`. Per-sample reports contain IDs and counts by default; `--include-raw-traces` explicitly enables local predicted arguments only. Aggregate reports contain no transcript/argument/state payloads. Ignore rules do not erase previously committed files: removing old EDA outputs from the current index preserves local files and leaves earlier Git history unchanged. No history rewrite or remote operation is performed by this refactor.
