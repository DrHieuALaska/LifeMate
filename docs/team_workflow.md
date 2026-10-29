# Five-person workflow

| Member | Primary ownership | Shared interface / handoff |
| --- | --- | --- |
| 1: speech | Local ASR, historical audio handling, transcription normalization | Implement ASR Protocol; produce current/history text without evaluator labels |
| 2: agent | Planning/controller, context use, termination and clarification | Implement Agent Protocol using allowed schemas and actual results |
| 3: model infrastructure | Local LLM serving, structured output, latency and resource budgets | Plug a model adapter into Agent.next_action; emit the fixed prediction JSONL |
| 4: tools/grounding | Catalog constraints, entity IDs, time interpretation and runtime loop | Own EnvRunner and Catalog; preserve one environment and complete tool logs |
| 5: evaluation/data | Locked manifests, metrics, stratification, privacy and experiment review | Own evaluator and failure taxonomy; publish safe aggregate findings |

Start with independent tests against synthetic requests. Run the fixed development benchmark for controller/model changes; keep prediction generation and scoring separate. Use a new run directory and record the exact configuration identifiers each time. Compare both total accuracy and relevant strata, including sample counts/uncertainty, rather than selecting on a favorable subgroup alone.

Coordinate interface changes through a PR reviewed by the adjacent owner. Metric changes require evaluation-owner review and a documented definition change. Never send a current gold call/result/state to a model to make a prediction look valid. History actions are already applied and must not be replayed. Rare-tool categories are derived from the training partition only.

Holdout IDs are read only for partition integrity during routine development. A separately authorized periodic holdout evaluation is outside this development CLI. Public-test inputs are not benchmark inputs, training material or a prompt-selection source. Keep recordings, text, states, manifests, private reports, prediction traces and model weights local under the designated ignored directories.

The first agent integration should consume the AgentInput dataclass, produce ToolDecision/Finish and export predictions. Reuse the evaluator unchanged. Before any real replay, ensure each evaluation run uses one fresh ToolEnv per sample and never retries from initial state. A valid call and a completed user task are different outcomes; review `done=false`, clarification stopping and changed-app metrics separately.
