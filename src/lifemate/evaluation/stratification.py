"""Overlapping retrospective strata. Features are evaluator-only."""
from lifemate.tools.catalog import has_constraint
from lifemate.utils.language import language_flags


def strata(record, catalog, training_frequency, frequency_available=True):
    calls = record['label']['tool_calls']
    names = [c['name'] for c in calls]
    tools = [catalog.tools.get(name, {}) for name in names]
    kinds = [t.get('kind', 'unknown') for t in tools]
    apps = {t.get('app_id', name.split('.')[0]) for t, name in zip(tools, names)}
    result = {'all', 'actions:'+str(len(calls)) if len(calls) < 4 else 'actions:4+',
              'apps:zero' if not apps else 'apps:single_including_sys' if len(apps) == 1 else 'apps:multi_including_sys'}
    business = apps-{'sys'}
    result.add('apps:zero_excluding_sys' if not business else 'apps:single_excluding_sys' if len(business) == 1 else 'apps:multi_excluding_sys')
    h = len(record.get('history', []))
    result.add('history:has' if h else 'history:none')
    if h:
        result.add('history:short_1_4' if h <= 4 else 'history:medium_5_12' if h <= 12 else 'history:long_13+')
    context = record.get('context', {})
    installed_n = len(context.get('installed_apps', {}))
    available_n = len(catalog.allowed(context))
    result.add('installed_apps:0_5' if installed_n <= 5 else 'installed_apps:6_10' if installed_n <= 10 else 'installed_apps:11+')
    result.add('available_tools:0_20' if available_n <= 20 else 'available_tools:21_40' if available_n <= 40 else 'available_tools:41+')
    if kinds and all(k == 'read' for k in kinds):
        result.add('kind:read_only')
    if 'write' in kinds:
        result.add('kind:contains_write')
    if 'meta' in kinds:
        result.add('kind:contains_meta')
    if any(k == 'read' and 'write' in kinds[i+1:] for i, k in enumerate(kinds)):
        result.add('kind:read_before_write')
    if len(names) != len(set(names)):
        result.add('kind:repeated_names')
    for constraint, label, val in [('x-ref', 'xref', None), ('x-one-of', 'one_of', None), ('x-match', 'fuzzy', 'fuzzy'), ('enum', 'enum', None)]:
        if any(has_constraint(t.get('parameters', {}), constraint, val) for t in tools):
            result.add('argument:contains_'+label)
    if any(key == 'id' or key.endswith(('_id', '_ids')) for c in calls for key in c['arguments']):
        result.add('argument:contains_id')
    if sum(len(t.get('parameters', {}).get('required', [])) for t in tools) >= 4:
        result.add('argument:high_required_4+')
    text = record.get('audio_text', '')
    for marker, flag in language_flags(text).items():
        if flag:
            result.add('language:'+marker)
    tokens = len(text.split())
    result.add('transcript:0_15' if tokens <= 15 else 'transcript:16_30' if tokens <= 30 else 'transcript:31+')
    for tool in tools:
        result.add('group:'+tool.get('group', 'unknown'))
        result.add('app:'+tool.get('app_id', 'unknown'))
    for name in names:
        safe_name = name if name in catalog.tools else '<unknown>'
        result.add('tool:'+safe_name)
        if not frequency_available:
            result.add('frequency:unavailable_without_train_manifest')
        else:
            count = training_frequency[name]
            result.add('frequency:unseen_0' if count == 0 else 'frequency:rare_1_5' if count <= 5 else 'frequency:medium_6_19' if count < 20 else 'frequency:common_20+')
    result.add('first_kind:'+kinds[0] if kinds else 'first_kind:none')
    return sorted(result)
