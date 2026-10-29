"""Write aggregate and default-redacted sample diagnostics to ignored artifacts only."""
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from lifemate.evaluation.metrics import prf, ratio


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_rows(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def reports(output, metrics, rows, strata, catalog, execution_mode=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'metrics.json', metrics)
    with (output/'per_sample.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    csv_rows(output/'stratified_metrics.csv', strata, ['stratum', 'n', 'small_sample', 'exact_trace_accuracy', 'wilson95_low', 'wilson95_high', 'tool_name_sequence_accuracy', 'tool_multiset_f1', 'argument_object_accuracy', 'argument_objects_evaluated', 'valid_predictions', 'primary_error_counts'])
    errors = metrics['overall']['error_counts']
    primary = metrics['overall']['primary_error_counts']
    csv_rows(output/'error_taxonomy.csv', [{'error': label, 'samples': count, 'percent': 100*count/len(rows), 'primary_samples': primary.get(label, 0)} for label, count in sorted(errors.items())], ['error', 'samples', 'percent', 'primary_samples'])
    cooccur = Counter(pair for row in rows for pair in itertools.combinations(sorted(row['errors']), 2))
    csv_rows(output/'error_cooccurrences.csv', [{'error_a': a, 'error_b': b, 'samples': count} for (a, b), count in sorted(cooccur.items())], ['error_a', 'error_b', 'samples'])
    tool_counts = {}
    for row in rows:
        for name, counts in row['per_tool'].items():
            if name not in tool_counts:
                tool_counts[name] = Counter()
            tool_counts[name].update(counts)
            tool_counts[name]['gold_samples'] += counts['tp']+counts['fn'] > 0
    tool_rows = []
    for name in sorted(catalog.tools.keys() | tool_counts.keys()):
        c = tool_counts.get(name, Counter())
        tool_rows.append({'tool': name, 'gold_samples': c['gold_samples'], 'small_sample': c['gold_samples'] < 20, **prf(c['tp'], c['fp'], c['fn'])})
    csv_rows(output/'per_tool_metrics.csv', tool_rows, ['tool', 'gold_samples', 'small_sample', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
    if execution_mode:
        exec_rows = [r['execution'] for r in rows if r.get('execution') is not None]
        state_rows = [r for r in exec_rows if r.get('changed_app_state_agreement') is not None]
        stopping = [r for r in exec_rows if r.get('stops_after_clarification') is not None]
        execution = {'samples_replayed': len(exec_rows), 'state_evaluable_samples': len(state_rows),
                     'changed_app_state_accuracy': ratio(sum(r['changed_app_state_agreement'] for r in state_rows), len(state_rows)),
                     'unintended_change_samples': sum(bool(r.get('unexpected_changed_apps')) for r in state_rows),
                     'clarification_stopping_evaluable_samples': len(stopping),
                     'stops_after_clarification_rate': ratio(sum(r['stops_after_clarification'] for r in stopping), len(stopping)),
                     **{key: sum(r.get(key, 0) for r in exec_rows) for key in ('attempted_calls', 'accepted_calls', 'rejected_calls', 'done_false_calls', 'waiting_for_user_calls', 'runtime_exceptions')},
                     'runtime_call_validity': ratio(sum(r['accepted_calls'] for r in exec_rows), sum(r['attempted_calls'] for r in exec_rows)),
                     'limitation': 'Strict raw changed-app comparison and runtime acceptance diagnostics; not official scoring or proven task completion.'}
        write_json(output/'execution_summary.json', execution)
    overall = metrics['overall']
    def show(value):
        return 'N/A' if value is None else f'{value:.6f}'
    text = f"""# Development benchmark: {metrics['provenance']['experiment_name']}

Prediction/model metadata are user-supplied provenance, not a claim of an AI model result.

Evaluated samples: **{overall['n']}**. Full development manifest: **{metrics['manifest_validation']['full_dev_count']}**. Subset run: **{metrics['manifest_validation']['is_subset']}**.

| Metric | Value |
| --- | --- |
| Valid prediction coverage | {metrics['coverage']['valid']}/{overall['n']} |
| Exact trace accuracy | {show(overall['exact_trace_accuracy'])} |
| Tool-name sequence accuracy | {show(overall['tool_name_sequence_accuracy'])} |
| Micro tool multiset F1 | {show(overall['tool_multiset']['f1'])} |
| Aligned argument-object accuracy | {show(overall['argument_object_accuracy'])} |
| Mean normalized name edit distance (valid predictions only) | {show(overall['normalized_edit_distance_mean'])} |

Exact/name accuracy includes missing and malformed predictions as failures. Argument accuracy includes aligned identical tool names only; inspect its denominator in metrics.json. Micro tool F1 counts occurrences and ignores order. No-action and clarification precision/recall/F1, Wilson intervals, per-tool metrics and all denominators are recorded in metrics.json/CSV files. Missing predictions are abstentions, never implicit empty traces.

Strata overlap; do not sum subgroup counts. Groups with N < 20 are marked small-sample. Linguistic markers and read-before-write are heuristics, not verified intent/dependency labels. x-ref differences indicate gold disagreement, not proven invalid IDs. Possible duplicate writes are review flags. No ASR root cause is inferred.

## Primary error counts

"""
    text += '\n'.join(f'- {key}: {value}' for key, value in sorted(primary.items()))+'\n'
    text += '\nAll sample traces, states, transcripts and personal argument values are excluded from this aggregate report. Per-sample output is redacted by default; raw predicted calls require an explicit local debugging flag. Execution metrics, when enabled, do not reproduce verified official scoring.\n'
    (output/'summary.md').write_text(text, encoding='utf-8')
