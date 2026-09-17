"""Linux process-group watchdog for future authorized v2 runs.

RSS is sampled: fast allocation can overshoot between samples. This is not an
OS memory reservation/cgroup and does not exempt jobs from desktop protections.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Sequence


def _group_rss_bytes(pgid: int) -> int:
    total = 0
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            # The comm field can contain spaces and parentheses.
            fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:  # field 5: pgrp; tail begins at field 3
                total += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def _terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Also reap/kill grandchildren if the leader exited first.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(command: Sequence[str], *, max_rss_mb: float, timeout_seconds: float,
              cwd: Path | None = None, env: dict[str, str] | None = None) -> dict:
    """Run one isolated worker, cap sampled group RSS/time, preserve bounded output."""
    if not sys.platform.startswith('linux') or not Path('/proc/self/stat').is_file():
        raise RuntimeError('v2 resource supervision requires Linux /proc; no unguarded fallback')
    for value in (max_rss_mb, timeout_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite resource budget required')
    start, peak = time.monotonic(), 0
    with tempfile.TemporaryFile() as stream:
        process = subprocess.Popen(list(command), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                peak = max(peak, _group_rss_bytes(process.pid))
                if peak > max_rss_mb * 1024 ** 2:
                    raise MemoryError(f'worker RSS exceeded {max_rss_mb} MiB budget')
                if time.monotonic() - start > timeout_seconds:
                    raise TimeoutError(f'worker timeout exceeded {timeout_seconds} seconds')
                time.sleep(0.05)
            # A worker must not detach running descendants and report success.
            if _group_rss_bytes(process.pid):
                raise RuntimeError('worker left active descendants')
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16384))
            output = stream.read().decode('utf-8', errors='replace')
            if process.returncode:
                raise RuntimeError(f'worker exited {process.returncode}: {output}')
            return {'returncode': process.returncode, 'output': output,
                    'peak_rss_mb': peak / 1024 ** 2, 'wall_seconds': time.monotonic() - start}
        finally:
            if process.poll() is None or _group_rss_bytes(process.pid):
                _terminate_group(process)
