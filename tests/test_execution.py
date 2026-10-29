import subprocess
import sys
import unittest
from pathlib import Path
from _support import FakeEnv, call, record, catalog
from lifemate.tools.env_runner import EnvRunner
from lifemate.evaluation.execution import summarize_execution
from lifemate.evaluation.dataset import build_agent_input
from lifemate.agent import Finish, ToolDecision


class ExecutionTests(unittest.TestCase):
    def test_one_instance_per_sample_no_reset(self):
        created = []
        runner = EnvRunner(lambda sid: created.append(sid) or FakeEnv())
        runner.replay('synthetic', [call(x=1)])
        with self.assertRaises(RuntimeError): runner.replay('synthetic', [])
        self.assertEqual(created, ['synthetic'])

    def test_failed_calls_stay_logged(self):
        result = EnvRunner(lambda sid: FakeEnv()).replay('synthetic', [call(x=-1), call(x=2)])
        self.assertEqual(len(result.runtime_calls), 2)
        summary = summarize_execution(result, {'a': {'x': 2}})
        self.assertEqual(summary['rejected_calls'], 1)
        self.assertTrue(summary['changed_app_state_agreement'])

    def test_ok_true_not_completion(self):
        obs = EnvRunner(lambda sid: FakeEnv()).replay('synthetic', [call(x=0)])
        self.assertEqual(summarize_execution(obs, {})['done_false_calls'], 1)

    def test_unintended_change(self):
        obs = EnvRunner(lambda sid: FakeEnv()).replay('synthetic', [call(x=1), call('b.write')])
        summary = summarize_execution(obs, {'a': {'x': 1}})
        self.assertFalse(summary['changed_app_state_agreement'])
        self.assertEqual(summary['unexpected_changed_apps'], 1)

    def test_exception_no_retry(self):
        obs = EnvRunner(lambda sid: FakeEnv()).replay('synthetic', [call(x=99), call(x=1)])
        self.assertEqual(obs.exception_count, 1); self.assertEqual(len(obs.attempted_calls), 1)

    def test_constructor_failure_is_consumed(self):
        def broken(sid): raise RuntimeError('synthetic constructor failure')
        runner = EnvRunner(broken)
        with self.assertRaises(RuntimeError): runner.replay('synthetic', [])
        with self.assertRaisesRegex(RuntimeError, 'already attempted'): runner.replay('synthetic', [])

    def test_hash_failure_preserves_attempt_log(self):
        class HashFailure(FakeEnv):
            def get_state_hash(self): raise RuntimeError('synthetic hash failure')
        observation = EnvRunner(lambda sid: HashFailure()).replay('synthetic', [call(x=1)])
        self.assertEqual(len(observation.attempted_calls), 1)
        self.assertEqual(len(observation.runtime_calls), 1)
        self.assertEqual(observation.exception_count, 1)
        self.assertIsNone(observation.state_hash)

    def test_clarification_stopping(self):
        obs = EnvRunner(lambda sid: FakeEnv()).replay('synthetic', [call('sys.ask_missing_info'), call(x=1)])
        self.assertFalse(summarize_execution(obs)['stops_after_clarification'])

    def test_live_interface_has_results_no_labels(self):
        class SyntheticDecisionAdapter:
            def __init__(self): self.seen = []
            def next_action(self, request):
                self.seen.append(request)
                return Finish() if request.previous_tool_results else ToolDecision('a.read', {})
        agent = SyntheticDecisionAdapter()
        obs = EnvRunner(lambda sid: FakeEnv()).run_agent('synthetic', agent, build_agent_input(record(), catalog()))
        self.assertTrue(obs.terminated); self.assertEqual(len(obs.attempted_calls), 1)
        self.assertEqual(len(agent.seen[1].previous_tool_results), 1)
        self.assertTrue(all(not hasattr(request, 'label') for request in agent.seen))

    def test_guard_blocks_direct_environment_and_public_reads(self):
        src = str(Path(__file__).resolve().parents[1]/'src')
        code = ('import sys; sys.path.insert(0,'+repr(src)+'); from lifemate.utils.safety import benchmark_guard\n'
                'with benchmark_guard():\n'
                ' for path in ["virtual/train_environment/x/state.json", "virtual/public_test/public_test.jsonl"]:\n'
                '  try: open(path)\n'
                '  except PermissionError: pass\n'
                '  else: raise AssertionError("access was not blocked")\n')
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)
