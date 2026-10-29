"""Name-first deterministic alignment and exact JSON argument diagnostics."""
from collections import Counter

from lifemate.tools.catalog import has_constraint, time_parameter
from lifemate.utils.json import equal, kind


def align_names(gold, predicted):
    """Levenshtein alignment. Ties: diagonal (match/substitute), delete, insert."""
    m, n = len(gold), len(predicted)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j-1]+(gold[i-1] != predicted[j-1]), dp[i-1][j]+1, dp[i][j-1]+1)
    i, j, operations = m, n, []
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(gold[i-1] != predicted[j-1]):
            operations.append(('match' if gold[i-1] == predicted[j-1] else 'substitute', i-1, j-1))
            i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j]+1:
            operations.append(('delete', i-1, None)); i -= 1
        else:
            operations.append(('insert', None, j-1)); j -= 1
    return dp[m][n], list(reversed(operations))


def diff_values(gold, predicted):
    """Nested mismatch counts without retaining values or sensitive key paths."""
    out = Counter()
    if kind(gold) != kind(predicted):
        out['type_mismatches'] += 1
        out['value_mismatches'] += 1
    elif isinstance(gold, dict):
        out['missing_keys'] += len(gold.keys()-predicted.keys())
        out['extra_keys'] += len(predicted.keys()-gold.keys())
        for key in sorted(gold.keys() & predicted.keys()):
            out.update(diff_values(gold[key], predicted[key]))
    elif isinstance(gold, list):
        out['missing_array_items'] += max(len(gold)-len(predicted), 0)
        out['extra_array_items'] += max(len(predicted)-len(gold), 0)
        for a, b in zip(gold, predicted):
            out.update(diff_values(a, b))
    elif not equal(gold, predicted):
        out['value_mismatches'] += 1
    return out


def argument_counts(gold, predicted, schema):
    out = Counter(diff_values(gold, predicted))
    out['argument_objects_total'] = 1
    out['argument_objects_correct'] = int(equal(gold, predicted))
    props, required = schema.get('properties', {}), set(schema.get('required', []))
    for key in sorted(set(gold) | set(predicted) | required):
        a, b = key in gold, key in predicted
        correct = a and b and equal(gold[key], predicted[key])
        categories = ['required' if key in required else 'optional']
        if a or b:
            categories.append('all_parameters')
        spec = props.get(key, {})
        if a or b:
            for constraint, category, value in [('enum', 'enum', None), ('x-ref', 'xref', None), ('x-match', 'fuzzy', 'fuzzy')]:
                if has_constraint(spec, constraint, value):
                    categories.append(category)
            if time_parameter(key, spec):
                categories.append('time')
        for category in categories:
            out[category+'_total'] += 1
            out[category+'_correct'] += int(correct)
    return out


def compare_traces(gold, predicted, catalog):
    gnames = [c['name'] for c in gold]
    pnames = [c['name'] for c in predicted]
    distance, alignment = align_names(gnames, pnames)
    gc, pc = Counter(gnames), Counter(pnames)
    per_tool = {name: {'tp': min(gc[name], pc[name]), 'fp': max(pc[name]-gc[name], 0),
                       'fn': max(gc[name]-pc[name], 0)} for name in sorted(gc.keys() | pc.keys())}
    args, operations = Counter(), Counter()
    for op, i, j in alignment:
        operations[op] += 1
        if op == 'match':
            schema = catalog.tools.get(gold[i]['name'], {}).get('parameters', {})
            args.update(argument_counts(gold[i]['arguments'], predicted[j]['arguments'], schema))
    return {'exact': gnames == pnames and all(equal(a['arguments'], b['arguments']) for a, b in zip(gold, predicted)),
            'names_exact': gnames == pnames, 'edit_distance': distance,
            'normalized_edit_distance': distance/max(len(gold), len(predicted), 1),
            'order_mismatch': gc == pc and gnames != pnames,
            'operations': dict(operations), 'arguments': dict(args), 'per_tool': per_tool,
            'gold_calls': len(gold), 'predicted_calls': len(predicted),
            'tp': sum(x['tp'] for x in per_tool.values()),
            'fp': sum(x['fp'] for x in per_tool.values()),
            'fn': sum(x['fn'] for x in per_tool.values())}
