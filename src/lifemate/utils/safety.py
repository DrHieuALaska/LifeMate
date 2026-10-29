"""Explicit-path access. Only the official ToolEnv constructor may load an environment."""
import contextlib
import contextvars
import os
import sys
from pathlib import Path

_runtime_access = contextvars.ContextVar('official_runtime_access', default=False)
_benchmark_active = contextvars.ContextVar('benchmark_active', default=False)
_installed = False
AUDIT = {'blocked_environment_events': 0, 'blocked_public_test_events': 0,
         'official_environment_open_events': 0}


def environment_path(path):
    try:
        parts = Path(os.fsdecode(path)).parts
    except TypeError:
        return False
    return any(p.casefold().endswith('_environment') or p.casefold() == 'state.json' for p in parts)


def public_path(path):
    try:
        return any(p.casefold() == 'public_test' for p in Path(os.fsdecode(path)).parts)
    except TypeError:
        return False


def safe_path(path):
    if environment_path(path) or public_path(path):
        raise PermissionError('Benchmark paths must not reference environment files or public_test')
    resolved = Path(path).resolve()
    if environment_path(resolved) or public_path(resolved):
        raise PermissionError('Benchmark path resolves to restricted data')
    return resolved


def install_audit_guard():
    global _installed
    if _installed:
        return
    def audit(event, args):
        if not _benchmark_active.get() or event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        if public_path(args[0]):
            AUDIT['blocked_public_test_events'] += 1
            raise PermissionError('Public-test access is forbidden in benchmark runs')
        if environment_path(args[0]):
            if not _runtime_access.get():
                AUDIT['blocked_environment_events'] += 1
                raise PermissionError('Direct environment access is forbidden')
            if event == 'open':
                AUDIT['official_environment_open_events'] += 1
    sys.addaudithook(audit)
    _installed = True


@contextlib.contextmanager
def benchmark_guard():
    """Scope the installed audit hook to this benchmark, not unrelated EDA/test work."""
    install_audit_guard()
    token = _benchmark_active.set(True)
    try:
        yield
    finally:
        _benchmark_active.reset(token)


@contextlib.contextmanager
def official_runtime_access():
    token = _runtime_access.set(True)
    try:
        yield
    finally:
        _runtime_access.reset(token)
