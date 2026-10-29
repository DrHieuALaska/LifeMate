"""Schema diagnostics. These checks do not claim official-validator equivalence."""
import datetime as dt
import re
from copy import deepcopy

from lifemate.utils.json import equal, kind, loads
from lifemate.utils.safety import safe_path

SYSTEM_TOOL = 'sys.ask_missing_info'  # Documented as always available in zaic_tools/README.md.


def has_constraint(schema, key, value=None):
    if isinstance(schema, dict):
        if key in schema and (value is None or schema[key] == value):
            return True
        return any(has_constraint(v, key, value) for v in schema.values())
    if isinstance(schema, list):
        return any(has_constraint(v, key, value) for v in schema)
    return False


def empty(value):
    return value is None or value == '' or (isinstance(value, (dict, list)) and not value)


def time_parameter(name, schema):
    return schema.get('format') in ('date', 'date-time', 'time') or bool(
        set(re.split(r'[_\W]+', name.casefold())) & {'time', 'date', 'datetime', 'start', 'end', 'due', 'at'})


def schema_errors(arguments, schema):
    if not isinstance(arguments, dict):
        return ['object_type']
    errors = []
    props = schema.get('properties', {})
    for key in schema.get('required', []):
        if key not in arguments or empty(arguments[key]):
            errors.append('missing_required')
    alternatives = schema.get('x-one-of', [])
    if alternatives and not any(all(k in arguments and not empty(arguments[k]) for k in group) for group in alternatives):
        errors.append('x_one_of')
    for key, value in arguments.items():
        if key not in props:
            if schema.get('additionalProperties') is False:
                errors.append('extra_forbidden')
            continue
        if empty(value):
            continue  # The official runtime ignores empty optional values.
        errors.extend(_value_errors(value, props[key]))
    return errors


def _value_errors(value, schema):
    errors = []
    tag = kind(value)
    expected = schema.get('type')
    if expected:
        valid = (tag == expected or expected == 'integer' and isinstance(value, int) and not isinstance(value, bool))
        if not valid:
            return ['type']
    if 'enum' in schema and not any(equal(value, x) for x in schema['enum']):
        errors.append('enum')
    if tag == 'number':
        for key, fails in [('minimum', lambda x: value < x), ('maximum', lambda x: value > x),
                           ('exclusiveMinimum', lambda x: value <= x), ('exclusiveMaximum', lambda x: value >= x)]:
            if key in schema and fails(schema[key]):
                errors.append('range')
    if schema.get('format') in ('date', 'date-time', 'time'):
        try:
            fmt = schema['format']
            if not isinstance(value, str):
                raise ValueError
            if fmt == 'date':
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                    raise ValueError
                dt.date.fromisoformat(value)
            elif fmt == 'date-time':
                if 'T' not in value:
                    raise ValueError
                dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                dt.time.fromisoformat(value)
        except ValueError:
            errors.append('format')
    if tag == 'array' and isinstance(schema.get('items'), dict):
        for item in value:
            errors.extend(_value_errors(item, schema['items']))
    if tag == 'object':
        errors.extend(schema_errors(value, schema))
    return errors


class Catalog:
    def __init__(self, data):
        self.tools = {}
        for app in data.get('apps', []):
            for tool in app.get('tools', []):
                name = tool['name']
                if name in self.tools:
                    raise ValueError('Duplicate tool in catalog')
                self.tools[name] = {**deepcopy(tool), 'app_id': app['id'], 'group': app.get('group', 'unknown')}

    @classmethod
    def from_path(cls, path):
        return cls(loads(safe_path(path).read_text(encoding='utf-8-sig')))

    def allowed(self, context):
        apps = context.get('installed_apps', {})
        names = {name for spec in apps.values() for name in spec.get('tools', [])}
        if SYSTEM_TOOL in self.tools:
            names.add(SYSTEM_TOOL)
        return names

    def schemas(self, context):
        return tuple(deepcopy(self.tools[n]) for n in sorted(self.allowed(context)) if n in self.tools)

    def validate(self, call, allowed):
        name = call['name']
        if name not in self.tools:
            return {'unknown_tool'}
        result = set(schema_errors(call['arguments'], self.tools[name].get('parameters', {})))
        if name not in allowed:
            result.add('unavailable_tool')
        return result
