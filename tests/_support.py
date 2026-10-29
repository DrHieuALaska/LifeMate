import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import copy
import json

from lifemate.tools.catalog import Catalog

CATALOG_DATA = {'apps': [
    {'id': 'a', 'group': 'home', 'tools': [
        {'name': 'a.read', 'kind': 'read', 'parameters': {'properties': {}}},
        {'name': 'a.write', 'kind': 'write', 'parameters': {'type': 'object', 'properties': {
            'x': {'type': 'integer', 'minimum': 0}, 'ref': {'type': 'string', 'x-ref': 'a.items'},
            'text': {'type': 'string', 'x-match': 'fuzzy'}, 'when': {'type': 'string', 'format': 'date-time'},
            'mode': {'type': 'string', 'enum': ['on', 'off']}}, 'required': ['x']}},
        {'name': 'a.choice', 'kind': 'write', 'parameters': {'properties': {'ref': {'type': 'string'}, 'other': {'type': 'string'}}, 'x-one-of': [['ref'], ['other']]}}
    ]},
    {'id': 'b', 'group': 'communication', 'tools': [
        {'name': 'b.write', 'kind': 'write', 'parameters': {'properties': {}}}]},
    {'id': 'sys', 'group': 'system', 'tools': [
        {'name': 'sys.ask_missing_info', 'kind': 'meta', 'parameters': {'properties': {}}}]}
]}


def catalog():
    return Catalog(CATALOG_DATA)


def call(name='a.write', **arguments):
    return {'name': name, 'arguments': arguments}


def record(sid='synthetic_1', calls=None, history=0):
    return {'id': sid, 'audio_text': 'SYNTHETIC_PRIVATE_VALUE', 'history': [{'role': 'user', 'audio_text': 'synthetic history'} for _ in range(history)],
            'context': {'installed_apps': {'a': {'tools': ['a.read', 'a.write', 'a.choice']}, 'b': {'tools': ['b.write']}},
                        'user_profile': {'name': 'SYNTHETIC_PRIVATE_VALUE'}, 'location': {}},
            'label': {'tool_calls': [] if calls is None else calls, 'tool_results': [{'ok': True, 'secret': 'GOLD_RESULT_SECRET'}], 'final_state': {}}}


def fixture(root):
    data = root/'data'
    (data/'train').mkdir(parents=True)
    split = data/'manifests'
    split.mkdir()
    records = [record('dev_1', [call(x=1)]), record('dev_2'), record('train_1', [call('a.read')]), record('holdout_1', [call('b.write')])]
    (data/'train/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    (data/'train/train_tools_catalog.json').write_text(json.dumps(CATALOG_DATA), encoding='utf-8')
    for name, ids in [('dev', ['dev_2', 'dev_1']), ('train', ['train_1']), ('holdout', ['holdout_1'])]:
        (split/(name+'_ids.txt')).write_text('\n'.join(ids)+'\n', encoding='utf-8')
    return data, split/'dev_ids.txt'


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.state = {'a': {'x': 0}, 'b': {'x': 0}}

    def call(self, name, arguments):
        self.calls.append({'name': name, 'arguments': copy.deepcopy(arguments)})
        if arguments.get('x') == 99:
            raise RuntimeError('Synthetic failure')
        if arguments.get('x') == -1:
            return {'ok': False, 'error': 'synthetic'}
        if name == 'sys.ask_missing_info':
            return {'ok': True, 'waiting_for_user': True}
        if name == 'a.write':
            if arguments['x'] == 0:
                return {'ok': True, 'done': False}
            self.state['a']['x'] = arguments['x']
        if name == 'b.write':
            self.state['b']['x'] = 1
        return {'ok': True}

    def get_tool_calls(self):
        return copy.deepcopy(self.calls)

    def get_state(self):
        return copy.deepcopy(self.state)

    def get_state_hash(self):
        return 'synthetic-hash'
