"""Evaluator-only result/state diagnostics. Never serialize snapshots or result payloads."""
from lifemate.utils.json import equal


def summarize_execution(observation, gold_final_state=None):
    results = observation.actual_results
    ok = sum(isinstance(r, dict) and r.get('ok') is True for r in results)
    rejected = sum(isinstance(r, dict) and r.get('ok') is False for r in results)
    done_false = sum(isinstance(r, dict) and r.get('done') is False for r in results)
    waiting = [i for i, r in enumerate(results) if isinstance(r, dict) and r.get('waiting_for_user') is True]
    result = {'attempted_calls': len(observation.attempted_calls), 'accepted_calls': ok,
              'rejected_calls': rejected, 'unknown_ok_calls': len(results)-ok-rejected,
              'done_false_calls': done_false,
              'waiting_for_user_calls': len(waiting), 'runtime_failures': max(rejected, observation.exception_count),
              'runtime_exceptions': observation.exception_count,
              'runtime_log_consistent': equal(observation.attempted_calls, observation.runtime_calls),
              'state_hash_available': isinstance(observation.state_hash, str),
              'stops_after_clarification': not any(i < len(results)-1 for i in waiting) if waiting else None,
              'agent_explicit_finish': observation.terminated,
              'changed_app_state_agreement': None, 'unexpected_changed_apps': None}
    if observation.before is not None and observation.after is not None and gold_final_state is not None:
        before, after = observation.before, observation.after
        changed = {app for app in before.keys() | after.keys() if app not in before or app not in after or not equal(before[app], after[app])}
        expected = set(gold_final_state)
        matching = sum(app in after and equal(after[app], gold_final_state[app]) for app in expected)
        result.update({'changed_app_state_agreement': changed == expected and matching == len(expected),
                       'unexpected_changed_apps': len(changed-expected), 'missing_changed_apps': len(expected-changed),
                       'expected_changed_apps': len(expected), 'observed_changed_apps': len(changed),
                       'matching_expected_apps': matching, 'no_action_state_unchanged': not changed if not observation.attempted_calls else None})
    return result
