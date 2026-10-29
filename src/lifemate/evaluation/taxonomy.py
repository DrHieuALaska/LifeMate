"""Observable errors and explicitly labeled suspicion; never infer an ASR root cause."""
from collections import Counter

from lifemate.tools.catalog import SYSTEM_TOOL
from lifemate.utils.json import canonical

PRECEDENCE = (
    'MISSING_PREDICTION', 'MALFORMED_PREDICTION', 'UNAVAILABLE_TOOL', 'INVALID_TOOL_SCHEMA',
    'RUNTIME_CALL_FAILURE', 'WRONG_NO_ACTION_DECISION', 'MISSED_MISSING_INFO', 'SPURIOUS_MISSING_INFO',
    'WRONG_TOOL_ORDER', 'WRONG_TOOL', 'MISSING_TOOL_CALL', 'EXTRA_TOOL_CALL',
    'ARGUMENT_TYPE_MISMATCH', 'MISSING_ARGUMENT', 'EXTRA_ARGUMENT', 'TIME_ARGUMENT_DISAGREEMENT',
    'XREF_ARGUMENT_DISAGREEMENT', 'ARGUMENT_MISMATCH', 'FINAL_STATE_MISMATCH', 'POSSIBLE_DUPLICATE_WRITE',
)


def classify(gold, predicted, comparison, catalog, allowed, coverage='valid', execution=None):
    labels = set()
    if coverage != 'valid':
        labels.add('MISSING_PREDICTION' if coverage == 'missing' else 'MALFORMED_PREDICTION')
    else:
        if bool(gold) != bool(predicted):
            labels.add('WRONG_NO_ACTION_DECISION')
        gmeta = any(c['name'] == SYSTEM_TOOL for c in gold)
        pmeta = any(c['name'] == SYSTEM_TOOL for c in predicted)
        if gmeta and not pmeta:
            labels.add('MISSED_MISSING_INFO')
        if pmeta and not gmeta:
            labels.add('SPURIOUS_MISSING_INFO')
        if comparison['order_mismatch']:
            labels.add('WRONG_TOOL_ORDER')
        for operation, label in [('substitute', 'WRONG_TOOL'), ('delete', 'MISSING_TOOL_CALL'), ('insert', 'EXTRA_TOOL_CALL')]:
            if comparison['operations'].get(operation, 0):
                labels.add(label)
        args = comparison['arguments']
        if args.get('argument_objects_correct', 0) < args.get('argument_objects_total', 0):
            labels.add('ARGUMENT_MISMATCH')
        for key, label in [('missing_keys', 'MISSING_ARGUMENT'), ('extra_keys', 'EXTRA_ARGUMENT'), ('type_mismatches', 'ARGUMENT_TYPE_MISMATCH')]:
            if args.get(key, 0):
                labels.add(label)
        for key, label in [('xref', 'XREF_ARGUMENT_DISAGREEMENT'), ('time', 'TIME_ARGUMENT_DISAGREEMENT')]:
            if args.get(key+'_correct', 0) < args.get(key+'_total', 0):
                labels.add(label)
        for call in predicted:
            errors = catalog.validate(call, allowed)
            if 'unavailable_tool' in errors:
                labels.add('UNAVAILABLE_TOOL')
            if errors-{'unavailable_tool'}:
                labels.add('INVALID_TOOL_SCHEMA')
            if 'missing_required' in errors:
                labels.add('MISSING_ARGUMENT')
        def call_key(c):
            return canonical({'name': c['name'], 'arguments': c['arguments']})
        gc = Counter(call_key(c) for c in gold)
        pc = Counter(call_key(c) for c in predicted)
        for call in predicted:
            key = call_key(call)
            if catalog.tools.get(call['name'], {}).get('kind') == 'write' and pc[key] >= 2 and pc[key] > gc[key]:
                labels.add('POSSIBLE_DUPLICATE_WRITE')
    if execution:
        if execution.get('runtime_failures', 0) or not execution.get('runtime_log_consistent', True):
            labels.add('RUNTIME_CALL_FAILURE')
        if execution.get('changed_app_state_agreement') is False:
            labels.add('FINAL_STATE_MISMATCH')
    ordered = [label for label in PRECEDENCE if label in labels]
    return ordered, ordered[0] if ordered else 'NONE'
