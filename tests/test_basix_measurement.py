"""Bounded measurement execution uses temporary CPU-only child processes."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("basix_measure_local", ROOT / "scripts" / "basix" / "measure_local.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


def test_bounded_process_returns_outputs_and_status():
    result = measure.run_bounded([sys.executable, "-c", "import sys; print('ok'); print('detail',file=sys.stderr); sys.exit(7)"], timeout=3)
    assert result.returncode == 7
    assert result.stdout == b"ok\n"
    assert result.stderr == b"detail\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX measurement lane")
def test_timeout_stops_wrapper_and_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat"
    child_code = (
        "import pathlib,sys,time; path=pathlib.Path(sys.argv[1]); "
        "\nwhile True:\n path.write_text(str(time.monotonic_ns())); time.sleep(.02)"
    )
    wrapper_code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2]]); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    child_pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            measure.run_bounded([sys.executable, "-c", wrapper_code, str(pid_file), str(heartbeat), child_code], timeout=1)
        child_pid = int(pid_file.read_text())
        stopped_value = heartbeat.read_text()
        time.sleep(.15)
        assert heartbeat.read_text() == stopped_value
        # Some Unix systems briefly retain a killed orphan as a zombie. Either
        # an absent PID or a zombie confirms it can no longer execute work.
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], text=True, capture_output=True, timeout=3).stdout.strip()
        assert not state or state.startswith("Z"), f"Measurement child still running: {state}"
    finally:
        # Test failure must not strand the synthetic child.
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_calibration_input_is_deterministic_png():
    value = measure.png_bytes()
    assert value == measure.png_bytes()
    assert value[:8] == b"\x89PNG\r\n\x1a\n"
