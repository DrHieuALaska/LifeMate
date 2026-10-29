"""Aggregate diagnostics with explicit denominators and Wilson intervals."""
import math
from collections import Counter


def ratio(n, d):
    return n/d if d else None


def prf(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': ratio(tp, tp+fp),
            'recall': ratio(tp, tp+fn), 'f1': ratio(2*tp, 2*tp+fp+fn)}


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p, denom = successes/n, 1+z*z/n
    center = (p+z*z/(2*n))/denom
    margin = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0, center-margin), min(1, center+margin)]


def aggregate(rows):
    n = len(rows)
    exact, names = sum(r['exact'] for r in rows), sum(r['names_exact'] for r in rows)
    counters = Counter()
    for r in rows:
        counters.update(r['arguments'])
    labels = Counter(label for r in rows for label in r['errors'])
    primary = Counter(r['primary_error'] for r in rows)
    defined_edits = [r['edit_distance'] for r in rows if r['edit_distance'] is not None]
    normalized = [r['normalized_edit_distance'] for r in rows if r['normalized_edit_distance'] is not None]
    result = {'n': n, 'small_sample': n < 20, 'valid_predictions': sum(r['coverage'] == 'valid' for r in rows),
              'exact_matches': exact, 'exact_trace_accuracy': ratio(exact, n), 'exact_trace_wilson95': wilson(exact, n),
              'tool_name_sequence_matches': names, 'tool_name_sequence_accuracy': ratio(names, n),
              'tool_name_wilson95': wilson(names, n),
              'tool_multiset': prf(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['fn'] for r in rows)),
              'sequence_edit_distance_total': sum(defined_edits), 'sequence_edit_distance_mean': ratio(sum(defined_edits), len(defined_edits)),
              'normalized_edit_distance_mean': ratio(sum(normalized), len(normalized)), 'edit_distance_evaluable_samples': len(defined_edits),
              'argument_counts': dict(counters), 'argument_object_accuracy': ratio(counters['argument_objects_correct'], counters['argument_objects_total']),
              'argument_metrics': {key: {'correct': counters[key+'_correct'], 'total': counters[key+'_total'], 'accuracy': ratio(counters[key+'_correct'], counters[key+'_total'])} for key in ('required', 'optional', 'all_parameters', 'enum', 'xref', 'fuzzy', 'time')},
              'error_counts': dict(labels), 'primary_error_counts': dict(primary)}
    for category in ('no_action', 'missing_info'):
        counts = [r[category] for r in rows]
        result[category] = prf(sum(x['tp'] for x in counts), sum(x['fp'] for x in counts), sum(x['fn'] for x in counts))
    return result


def stratified(rows):
    names = sorted({name for r in rows for name in r['strata']})
    result = []
    for name in names:
        value = aggregate([r for r in rows if name in r['strata']])
        result.append({'stratum': name, 'n': value['n'], 'small_sample': value['small_sample'],
                       'exact_trace_accuracy': value['exact_trace_accuracy'],
                       'wilson95_low': value['exact_trace_wilson95'][0], 'wilson95_high': value['exact_trace_wilson95'][1],
                       'tool_name_sequence_accuracy': value['tool_name_sequence_accuracy'],
                       'tool_multiset_f1': value['tool_multiset']['f1'],
                       'argument_object_accuracy': value['argument_object_accuracy'],
                       'argument_objects_evaluated': value['argument_counts'].get('argument_objects_total', 0),
                       'valid_predictions': value['valid_predictions'], 'primary_error_counts': value['primary_error_counts']})
    return result
