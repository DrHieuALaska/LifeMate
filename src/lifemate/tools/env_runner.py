"""Single-use official environments and a label-free future agent loop."""
import copy
import importlib.util
import sys
from dataclasses import dataclass, replace

from lifemate.agent import Finish, ToolDecision
from lifemate.utils.json import equal
from lifemate.utils.safety import official_runtime_access, safe_path


def official_factory(data_root):
    root = safe_path(data_root)
    package = root/'train/zaic_tools'
    if not (package/'__init__.py').is_file():
        raise ValueError('Missing documented train/zaic_tools runtime')
    # Load only this split under a private name, avoiding train/public module collisions.
    module_name = '_lifemate_official_train_tools'
    existing = sys.modules.get(module_name)
    if existing and safe_path(existing.__file__) != safe_path(package/'__init__.py'):
        raise ValueError('A different train runtime is already loaded in this process')
    if existing is None:
        spec = importlib.util.spec_from_file_location(module_name, package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = existing
    def factory(sid):
        # Only the official documented constructor may perform restricted file reads.
        with official_runtime_access():
            return module.ToolEnv('train', sid, root=root)
    return factory


@dataclass
class RunObservation:
    attempted_calls: list
    actual_results: list
    runtime_calls: list
    state_hash: str | None
    before: dict | None
    after: dict | None
    exception_count: int
    terminated: bool | None = None


class EnvRunner:
    def __init__(self, factory):
        self.factory = factory
        self._seen = set()

    def _create(self, sid):
        if sid in self._seen:
            raise RuntimeError('Sample environment already attempted in this run; reset/retry is forbidden')
        self._seen.add(sid)  # A constructor failure also cannot silently trigger a reset.
        return self.factory(sid)

    def replay(self, sid, calls, compare_state=True):
        return self._run(sid, iter(calls), compare_state)

    def _run(self, sid, decisions, compare_state, request=None, agent=None, max_calls=32):
        env = self._create(sid)
        attempted, results, errors, terminated = [], [], 0, None
        before = None
        if compare_state and callable(getattr(env, 'get_state', None)):
            try:
                before = copy.deepcopy(env.get_state())
            except Exception:
                errors += 1
        if agent:
            def live():
                nonlocal terminated
                for _ in range(max_calls+1):
                    decision = agent.next_action(replace(request, previous_tool_results=tuple(copy.deepcopy(results))))
                    if isinstance(decision, Finish):
                        terminated = True
                        return
                    if not isinstance(decision, ToolDecision) or not isinstance(decision.name, str) or not isinstance(decision.arguments, dict):
                        raise ValueError('Agent must return ToolDecision or Finish')
                    if len(attempted) >= max_calls:
                        terminated = False
                        return
                    yield {'name': decision.name, 'arguments': copy.deepcopy(decision.arguments)}
            decisions = live()
        for call in decisions:
            attempted.append(copy.deepcopy(call))  # Log every invocation before calling, including exceptions.
            try:
                result = env.call(call['name'], copy.deepcopy(call['arguments']))
            except Exception:
                errors += 1
                results.append({'ok': False, '_runner_exception': True})
                break  # No retries or fresh environments.
            results.append(copy.deepcopy(result))
        try:
            runtime_calls = env.get_tool_calls()
        except Exception:
            runtime_calls = []
            errors += 1
        try:
            state_hash = env.get_state_hash()
        except Exception:
            state_hash = None
            errors += 1
        after = None
        if before is not None:
            try:
                after = copy.deepcopy(env.get_state())
            except Exception:
                errors += 1
        return RunObservation(attempted, results, runtime_calls, state_hash, before, after, errors, terminated)

    def run_agent(self, sid, agent, request, max_calls=32):
        """Future integration: no labels or state snapshots enter an agent request."""
        return self._run(sid, (), False, request=request, agent=agent, max_calls=max_calls)
