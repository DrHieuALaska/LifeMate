"""Strict JSON parsing and equality, including boolean/number separation."""
import json
import math
from decimal import Decimal


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def loads(text):
    def reject(_):
        raise ValueError('Non-finite JSON number')
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)
    canonical(value)  # Reject overflowed float values such as 1e999.
    return value


def kind(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    raise ValueError('Not a JSON value')


def canonical(value):
    tag = kind(value)
    if tag == 'number':
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite JSON number')
        return tag, Decimal(str(value))
    if tag == 'object':
        if not all(isinstance(k, str) for k in value):
            raise ValueError('JSON object keys must be strings')
        return tag, tuple((k, canonical(v)) for k, v in sorted(value.items()))
    if tag == 'array':
        return tag, tuple(canonical(v) for v in value)
    return tag, value


def equal(a, b):
    return canonical(a) == canonical(b)
