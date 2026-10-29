# Contributing

Use focused feature branches such as `feature/asr-adapter`, `feature/agent-controller`, or `fix/trace-alignment`. Open a pull request describing the concrete behavior change, relevant tests, and reproducible validation commands. Have the interface owner and one adjacent-workstream reviewer assess shared changes.

Run `python -m unittest discover -s tests -v` before review. Fixtures must be synthetic and dataset-independent. Metric changes need tests for denominators, missing coverage, repeated calls and JSON types. Update docs when a protocol, schema, taxonomy or metric definition changes; do not silently compare experiments using incompatible definitions.

Keep current-request gold labels/results/states inside evaluator modules. An inference adapter must consume only the published agent/ASR interfaces and actual runtime results. Do not replay history or recreate an environment after an attempted action. Every env.call invocation, including rejection, belongs in the trace.

Keep each experiment in a new ignored artifacts directory. Record manifest checksum, Git revision, model/prompt/controller identifiers when known, and settings. Reuse the locked development IDs; routine work must not evaluate holdout or use public-test data to tune models/prompts.

Before staging, inspect `git status --short` and `git diff --cached --name-only`. Never use `git add -f` for competition data. Do not add prediction logs, private reports/manifests, audio, raw text, state files, model weights, credentials or environment configuration. No generated benchmark output belongs in a PR. An aggregate finding may be described in the PR without sensitive arguments or request examples.

Dependency additions should be small and justified. The core evaluator currently uses only the standard library; NumPy/Matplotlib are optional EDA extras. Preserve Python 3.11–3.14 and Windows/Linux path behavior. CI must run without GPUs, local models, secrets or private data.
