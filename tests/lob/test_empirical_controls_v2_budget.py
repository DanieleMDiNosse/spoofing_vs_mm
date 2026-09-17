"""Only tiny synthetic subprocesses; no market-data execution."""
import importlib.util
import sys

import pytest


def api():
    name = 'spoofing_detection.lob.empirical_controls_v2_budget'
    assert importlib.util.find_spec(name), 'resource supervisor missing'
    return __import__(name, fromlist=['supervise'])


def test_supervisor_returns_success_and_captured_output():
    result = api().supervise([sys.executable, '-c', 'print("synthetic-ok")'], max_rss_mb=512, timeout_seconds=10)
    assert result['returncode'] == 0
    assert 'synthetic-ok' in result['output']
    assert result['peak_rss_mb'] >= 0


def test_supervisor_times_out_and_reaps_worker():
    with pytest.raises(TimeoutError, match='timeout'):
        api().supervise([sys.executable, '-c', 'import time; time.sleep(30)'], max_rss_mb=512, timeout_seconds=0.15)


def test_supervisor_stops_on_memory_budget():
    with pytest.raises(MemoryError, match='RSS'):
        api().supervise([sys.executable, '-c', 'import time; data=bytearray(32*1024*1024); time.sleep(30)'], max_rss_mb=8, timeout_seconds=10)


def test_supervisor_nonzero_exit_is_not_success():
    with pytest.raises(RuntimeError, match='synthetic failure'):
        api().supervise([sys.executable, '-c', 'raise RuntimeError("synthetic failure")'], max_rss_mb=512, timeout_seconds=10)
