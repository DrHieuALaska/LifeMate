"""Train-only CLI and evaluator. Prediction generation is an independent boundary."""
import argparse
import datetime as dt
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from lifemate.evaluation.dataset import load_dev, load_predictions
from lifemate.evaluation.execution import summarize_execution
from lifemate.evaluation.metrics import aggregate, stratified
from lifemate.evaluation.reporting import reports
from lifemate.evaluation.stratification import strata
from lifemate.evaluation.taxonomy import classify
from lifemate.evaluation.traces import compare_traces
from lifemate.tools.catalog import SYSTEM_TOOL
from lifemate.tools.env_runner import EnvRunner, official_factory
from lifemate.utils.safety import AUDIT, benchmark_guard, install_audit_guard, safe_path


def binary_counts(actual, predicted):
    return {'tp': int(actual and predicted is True), 'fp': int(not actual and predicted is True),
            'fn': int(actual and predicted is not True)}


def evaluate(dataset, predictions, runner=None, compare_state=True, include_raw=False):
    rows = []
    catalog = dataset.catalog
    for record in dataset.records:
        sid, gold = record['id'], record['label']['tool_calls']
        coverage = 'malformed' if sid in predictions.malformed_ids else 'valid' if sid in predictions.calls else 'missing'
        predicted = predictions.calls.get(sid)
        if coverage == 'valid':
            comparison = compare_traces(gold, predicted, catalog)
        else:
            comparison = {'exact': False, 'names_exact': False, 'edit_distance': None,
                          'normalized_edit_distance': None, 'operations': {}, 'arguments': {},
                          'order_mismatch': False, 'gold_calls': len(gold), 'predicted_calls': None,
                          'per_tool': {name: {'tp': 0, 'fp': 0, 'fn': count} for name, count in Counter(c['name'] for c in gold).items()},
                          'tp': 0, 'fp': 0, 'fn': len(gold)}
        execution = None
        if runner is not None and coverage == 'valid':
            try:
                observation = runner.replay(sid, predicted, compare_state=compare_state)
                execution = summarize_execution(observation, record['label'].get('final_state') if compare_state else None)
            except Exception:
                # The claimed sample remains consumed; no retry. Never export exception messages.
                execution = {'runtime_failures': 1, 'runtime_exceptions': 1, 'runtime_log_consistent': False,
                             'attempted_calls': 0, 'accepted_calls': 0, 'rejected_calls': 0,
                             'changed_app_state_agreement': None, 'unexpected_changed_apps': None,
                             'stops_after_clarification': None, 'initialization_or_snapshot_failure': True}
        errors, primary = classify(gold, predicted, comparison, catalog, catalog.allowed(record.get('context', {})), coverage, execution)
        # Tool-name strings outside the catalog might themselves contain private text.
        safe_tools = {}
        for name, counts in comparison['per_tool'].items():
            safe_name = name if name in catalog.tools else '<unknown>'
            safe_tools.setdefault(safe_name, {'tp': 0, 'fp': 0, 'fn': 0})
            for key in counts:
                safe_tools[safe_name][key] += counts[key]
        comparison['per_tool'] = safe_tools
        row = {'id': sid, 'coverage': coverage, **comparison, 'errors': errors, 'primary_error': primary,
               'no_action': binary_counts(not gold, not predicted if coverage == 'valid' else None),
               'missing_info': binary_counts(any(c['name'] == SYSTEM_TOOL for c in gold),
                    any(c['name'] == SYSTEM_TOOL for c in predicted) if coverage == 'valid' else None),
               'strata': strata(record, catalog, dataset.training_tool_frequency, dataset.partition_validation['training_frequency_available']),
               'execution': execution}
        if include_raw and coverage == 'valid':
            row['raw_predicted_calls'] = predicted
        rows.append(row)
    return rows


def git_commit():
    try:
        result = subprocess.run(['git', '-c', 'safe.directory='+str(Path.cwd()), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty():
    try:
        result = subprocess.run(['git', '-c', 'safe.directory='+str(Path.cwd()), 'status', '--porcelain', '--untracked-files=normal'], capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


@benchmark_guard()
def run(data_root, manifest, predictions_path, output, mode='trace', experiment_name=None,
        model_id=None, prompt_id=None, controller_id=None, limit=None, compare_state=True,
        include_raw=False, expected_count=193, runner=None):
    install_audit_guard()
    root, manifest, predictions_path, output = map(safe_path, (data_root, manifest, predictions_path, output))
    # Benchmark output must live below an artifacts directory, outside confidential source splits.
    if 'artifacts' not in (p.casefold() for p in output.parts) or output.is_relative_to(root):
        raise ValueError('Benchmark outputs must be under an artifacts directory outside the dataset root')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new experiment directory')
    dataset = load_dev(root, manifest, expected_count=expected_count, limit=limit)
    predictions = load_predictions(predictions_path, dataset.dev_ids)
    if mode not in ('trace', 'replay'):
        raise ValueError('Mode must be trace or replay')
    if mode == 'replay' and runner is None:
        runner = EnvRunner(official_factory(root))
    rows = evaluate(dataset, predictions, runner if mode == 'replay' else None, compare_state, include_raw)
    coverage = Counter(r['coverage'] for r in rows)
    metrics = {'provenance': {'experiment_name': experiment_name or output.name,
                'dataset_path': str(root), 'manifest_path': str(manifest), 'manifest_sha256': dataset.manifest_checksum,
                'predictions_path': str(predictions_path), 'model_id': model_id, 'prompt_id': prompt_id,
                'controller_id': controller_id, 'git_commit': git_commit(), 'git_dirty': git_dirty(), 'mode': mode,
                'timestamp': dt.datetime.now(dt.timezone(dt.timedelta(hours=7), 'Asia/Bangkok')).isoformat(),
                'python_version': platform.python_version(), 'settings': {'limit': limit, 'state_comparison': 'raw_exact' if compare_state and mode == 'replay' else 'disabled', 'include_raw_predicted_calls': include_raw, 'alignment_tie_break': ['diagonal', 'delete', 'insert']}},
               'manifest_validation': dataset.partition_validation,
               'coverage': {key: coverage[key] for key in ('valid', 'missing', 'malformed')},
               'prediction_file_errors': predictions.file_errors, 'overall': aggregate(rows),
               'access_audit': dict(AUDIT)}
    metrics['coverage']['complete'] = coverage['valid'] == len(rows) and not predictions.file_errors
    result_strata = stratified(rows)
    reports(output, metrics, rows, result_strata, dataset.catalog, mode == 'replay')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate the fixed train-only development manifest. No model inference.')
    parser.add_argument('--config', help='JSON-compatible YAML 1.2 configuration; CLI values override it')
    parser.add_argument('--data-root', default='train+public_test')
    parser.add_argument('--manifest', default='train+public_test/reports/eda/validation_split/dev_ids.txt')
    parser.add_argument('--predictions', required=True, help='Local predicted tool-call JSONL')
    parser.add_argument('--output', required=True, help='New directory under artifacts/')
    parser.add_argument('--mode', choices=['trace', 'replay'], default='trace')
    parser.add_argument('--experiment-name')
    parser.add_argument('--model-id')
    parser.add_argument('--prompt-id')
    parser.add_argument('--controller-id')
    parser.add_argument('--limit', type=int, help='Evaluate the first N existing dev IDs; explicitly marked subset')
    parser.add_argument('--no-state-comparison', action='store_true')
    parser.add_argument('--include-raw-traces', action='store_true', help='Local debugging only: write predicted arguments under ignored artifacts')
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            config = json.loads(safe_path(preliminary.config).read_text(encoding='utf-8'))
            allowed = {action.dest for action in parser._actions}-{'help', 'config', 'predictions', 'output'}
            if not isinstance(config, dict) or set(config)-allowed:
                raise ValueError('Unknown config settings')
            parser.set_defaults(**config)
        except (OSError, ValueError) as exc:
            parser.error(f'Invalid JSON-compatible YAML config: {type(exc).__name__}')
    args = parser.parse_args(argv)
    try:
        result = run(args.data_root, args.manifest, args.predictions, args.output, args.mode,
                     args.experiment_name, args.model_id, args.prompt_id, args.controller_id, args.limit,
                     not args.no_state_comparison, args.include_raw_traces)
    except (ValueError, OSError) as exc:
        # Our validation messages are structural, never a copied source record or tool payload.
        parser.exit(2, f'Benchmark failed: {exc}\n')
    print(json.dumps({'evaluated_samples': result['overall']['n'], 'coverage': result['coverage'],
                      'exact_trace_accuracy': result['overall']['exact_trace_accuracy'],
                      'tool_name_sequence_accuracy': result['overall']['tool_name_sequence_accuracy'],
                      'tool_multiset_f1': result['overall']['tool_multiset']['f1']}, indent=2))
    return 0 if result['coverage']['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
