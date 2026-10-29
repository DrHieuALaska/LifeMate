# Train-only development evaluation protocol

## Inputs, ordering and coverage

The CLI requires the existing `dev_ids.txt` containing exactly 193 unique IDs. It streams only train/train.jsonl and train/train_tools_catalog.json, checks duplicate training IDs and unknown manifest IDs, and validates disjoint sibling train/dev/holdout manifests when available. Manifest order is the evaluation order. Holdout labels are never evaluated. A subset is the first N existing dev IDs and is explicitly marked; it is not reported as the full benchmark. Train-partition tool frequencies are accumulated only for training-manifest IDs. If the train manifest is absent, frequency strata are marked unavailable rather than estimated on dev.

Prediction JSONL objects require a string id and list tool_calls. Every call requires a nonempty string name and object arguments; state_hash and extra record metadata are ignored by trace comparison. Strict JSON rejects duplicate object keys, NaN/Infinity and overflowed nonfinite numbers. Empty lines are ignored. Invalid JSON without a reliable ID is a file-level error; its corresponding omitted sample remains missing. Duplicate IDs invalidate that ID, even if one line looks valid. Invalid call shapes become malformed predictions. Extra IDs and all file errors are reported; the CLI exits 2 for incomplete coverage/file errors while still writing diagnostic results. Structural/load failures exit 2 without claiming a completed benchmark.

Missing and malformed predictions count as failed exact/name outcomes using the full evaluated sample denominator. They are not converted to empty traces. Their gold occurrences contribute false negatives to micro tool metrics; no reliable predicted occurrences or edit/argument alignment is assumed. A missing prediction for a zero-call sample is still a failure. Output coverage is reported alongside all scores.

Outputs must be under an artifacts directory outside the dataset directory. A nonempty experiment directory is rejected, preventing accidental overwrite. Each run writes metrics.json, summary.md, stratified_metrics.csv, per_tool_metrics.csv, error_taxonomy.csv, error_cooccurrences.csv and per_sample.jsonl; replay also writes execution_summary.json.

## Trace metrics

1. **Exact trace accuracy:** equal call counts, ordered names and exact argument objects, divided by all evaluated samples. Dictionary order is irrelevant; array order matters. JSON strings, booleans, numbers, null, arrays and objects are distinct. JSON numbers 1 and 1.0 are equal; true and 1 are different. No case/whitespace/fuzzy normalization is applied to argument values.
2. **Tool-name sequence accuracy:** equal ordered tool-name lists, ignoring arguments, divided by all evaluated samples. Valid empty/empty traces match.
3. **Tool multiset precision/recall/F1:** for each sample/name, TP=min(gold count,predicted count), FP=max(predicted−gold,0), FN=max(gold−predicted,0). Sum counts across samples for micro scores. Repeated occurrences count separately. Per-tool metrics use these same counts, plus the number of gold-containing samples. Tools unknown to the catalog are aggregated as `<unknown>` to avoid leaking arbitrary predicted strings. Multiset scores ignore order and arguments and are diagnostic only.
4. **Name sequence edit distance:** Levenshtein costs are match=0 and substitution/insertion/deletion=1. Backtracking ties prefer diagonal, then deletion, then insertion. Normalize each valid sample by max(gold length,predicted length,1); report raw totals and raw/normalized means over valid predictions only, with their evaluable count. Equal multisets with different name sequences trigger WRONG_TOOL_ORDER. Argument content never influences name alignment. Repeated identical names may admit another equally optimal alignment; this deterministic alignment is one convention, not an oracle of intended correspondence.
5. **Argument agreement:** only aligned identical tool names contribute. Exact argument-object agreement uses aligned-call count as denominator. Missing/extra keys and mismatched leaf values/types are counted recursively; array length differences have separate counts. Required-parameter agreement includes every schema-required top-level key, including absent-required keys as failures. Optional agreement includes top-level keys in either argument object that are not schema-required. `all_parameters` uses the union of provided top-level keys. Enum/x-ref/fuzzy/time agreement uses constrained top-level keys present in either argument object; nested constraints mark the containing parameter. All absent-versus-present comparisons fail. Time fields use schema date/time format or exact name tokens time/date/datetime/start/end/due/at. These counts describe gold disagreement, not proven runtime invalidity. No tolerant fuzzy score is implemented or represented as official scoring.
6. **No-action and clarification recognition:** positive no-action means gold calls is empty; predicted positive means a valid empty prediction. Positive clarification means sys.ask_missing_info occurs anywhere in the trace, including after other calls. TP/FP/FN are sample-level. Missing/malformed predictions are abstentions and count FN for actual positives. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Missing/spurious clarification taxonomy labels apply to valid predictions; coverage errors remain separate for absent predictions.

Any zero denominator yields JSON null / blank CSV / N/A text, not an invented perfect score. For example all-empty traces have no occurrence-level F1 denominator, even when exact trace accuracy is 1. Wilson 95% intervals use z=1.959963984540054 for sample-level exact and name success rates. Intervals describe sample uncertainty and do not repair leakage or correlated examples. Strata mark N<20 as small-sample; no causal inference is made.

## Static validation and availability

The complete train catalog supplies required/optional arguments, types, enums, numeric bounds, formats, array schemas, x-one-of, x-ref and x-match metadata. The effective allowlist is installed-app tools plus the catalog's `sys.ask_missing_info`: the train runtime documentation explicitly makes this system tool always available. Keep explicit installed counts separate from effective permitted-tool counts.

Static diagnostics check known names, effective availability, nonempty required values, complete x-one-of groups, JSON types, enums, numeric bounds, date/time formats, array items and nested objects. Empty optional values are ignored following the documented runtime convention. Extra keys are rejected only when the schema explicitly forbids them. This independent validator is a preflight diagnostic; it has not been certified equivalent to every official-runtime quirk. It cannot validate whether an entity ID exists or a financial action can complete. x-ref mismatches against gold therefore receive a disagreement label, not an invented invalid-ID claim.

## Error taxonomy and primary precedence

The following ordered list is both the implemented taxonomy and deterministic primary-error precedence; all applicable labels are retained:

1. MISSING_PREDICTION
2. MALFORMED_PREDICTION
3. UNAVAILABLE_TOOL
4. INVALID_TOOL_SCHEMA
5. RUNTIME_CALL_FAILURE
6. WRONG_NO_ACTION_DECISION
7. MISSED_MISSING_INFO
8. SPURIOUS_MISSING_INFO
9. WRONG_TOOL_ORDER
10. WRONG_TOOL
11. MISSING_TOOL_CALL
12. EXTRA_TOOL_CALL
13. ARGUMENT_TYPE_MISMATCH
14. MISSING_ARGUMENT
15. EXTRA_ARGUMENT
16. TIME_ARGUMENT_DISAGREEMENT
17. XREF_ARGUMENT_DISAGREEMENT
18. ARGUMENT_MISMATCH
19. FINAL_STATE_MISMATCH
20. POSSIBLE_DUPLICATE_WRITE

WRONG_TOOL/MISSING_TOOL_CALL/EXTRA_TOOL_CALL follow name substitutions/deletions/insertions. WRONG_TOOL_ORDER requires equal name multisets but unequal order. ARGUMENT_MISMATCH means at least one aligned argument object differs; type/missing/extra/time/x-ref labels refine that observation. Missing schema-required parameters can also trigger MISSING_ARGUMENT/INVALID_TOOL_SCHEMA. A duplicate-write review flag requires an identical predicted write to occur at least twice and more often than the identical gold call; repeated gold operations are not automatically suspicious. A coverage problem can exist without an observable missed intent. Runtime rejection/exception/log inconsistency triggers runtime failure; raw changed-app disagreement triggers state mismatch. NONE means no taxonomy flag; exact trace accuracy remains an independent metric.

Counts and percentages use evaluated samples; labels overlap. Co-occurrence CSV counts sample membership in each pair of labels. Primary labels form a mutually exclusive partition. No ASR_ERROR or inferred semantic root cause is assigned from traces alone.

## Overlapping strata and fixed boundaries

| Dimension | Groups / definition |
| --- | --- |
| Gold action count | 0, 1, 2, 3, 4+ calls |
| History | none, has; nonempty short 1–4, medium 5–12, long 13+ historical turns |
| App complexity | zero/single/multiple distinct tool apps including sys; separately zero/single/multiple excluding sys |
| Installed apps | 0–5, 6–10, 11+ context app entries |
| Effective available tools | 0–20, 21–40, 41+ including always-available system tool |
| Tool kinds | nonempty read-only; contains write; contains meta; read before a later write; repeated names |
| Schema complexity | any gold-used tool schema contains x-ref/x-one-of/fuzzy/enum; any supplied id/_id/_ids argument; ≥4 summed schema-required parameters across calls |
| Language | unchanged EDA regex flags: relative time, clock time, date, digits, quantity, contact, negation, correction, conjunction, reference, English indicators |
| Transcript length | 0–15, 16–30, 31+ whitespace tokens |
| Families | each gold app, catalog app group, gold tool, first-action kind or none |
| Frequency | tool occurrence count in training-manifest records: unseen 0, rare 1–5, medium 6–19, common 20+; a sample may belong to several |

Every nonempty subgroup gets N, valid prediction count, exact accuracy/interval, name accuracy, micro multiset F1, aligned argument agreement/denominator and primary errors. Gold-derived strata stay in evaluator modules. Schema presence does not prove a constrained optional argument was supplied; lexical reference/conjunction markers do not prove history dependence or multiple intent. Read-before-write does not prove a result dependency. Overlapping subgroup Ns must not be summed. Action buckets are exclusive and cover the evaluated set.

## Execution and future live mode

Trace mode never imports the official runtime or reads audio/environment/public-test files. Optional replay creates a single documented train ToolEnv per valid sample, executes only predicted calls in order, records every invocation (including failed calls), obtains the official runtime log and state hash, and never resets/retries or runs gold actions. Snapshot comparison uses only env.get_state() in evaluator memory. If snapshots are disabled/unavailable, state metrics are null. Original state files are never directly opened by our code.

Raw exact comparison detects changed apps from before/after snapshots, compares them with label.final_state app/value expectations, and counts unexpected or missing changed apps. No state values enter outputs or an agent. This is deliberately separate from official canonicalization/fuzzy call scoring and is not a reproduction of official scoring.

Runtime acceptance is ok=true divided by actual attempted calls. Rejected calls, exceptions, done=false and waiting_for_user are separate diagnostics. A missing done field does not prove completion. Clarification stopping is measurable only when waiting_for_user appears: it is valid under this diagnostic if no later call occurs. General correct stopping/task completion cannot be inferred solely from a finite trace. The future live loop records explicit Finish or budget exhaustion separately.

Real official replay is optional and not part of the state-free verification run. Synthetic fake-environment tests exercise creation limits, logging, rejection, exceptions, state mismatches, unintended changes and stopping. No real ASR/LLM is implemented. AgentInput contains current transcript, sanitized copies of legitimate context/history, allowed schemas and actual prior runtime results; current gold labels/results/states are excluded. History actions have already changed the initial state and must not be replayed.

## Reproducibility and privacy

Each run records experiment name, dataset/manifest/prediction paths, SHA256 of manifest bytes, full/evaluated counts, supplied model/prompt/controller identifiers or null, Git revision when available, execution mode, timestamp with +07:00 offset, Python version and evaluator settings. The checksum is byte-based, so CRLF/LF changes are visible even if ID membership is unchanged. No metadata is invented.

Metrics and Markdown summaries exclude request text, user-profile/location values, tool arguments, result payloads and app states. Default per_sample.jsonl contains IDs, diagnostic counts, labels, strata and safe execution summaries. `--include-raw-traces` explicitly adds predicted calls/arguments only under ignored artifacts. Runtime snapshots and gold results are never serialized. Core tests are synthetic, including synthetic privacy sentinels, and run without the competition package/GPU. Actual no-action smoke metrics must be labeled as a deliberately trivial baseline, not model accuracy.
