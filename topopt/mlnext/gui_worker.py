"""Background worker launched by the local GUI; invoked via python -m.

This is a local subprocess, not an online cloud job. It records all attempts,
captures live output, handles cancellation, and survives Streamlit reruns.
"""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .gui_backend import ROOT, RUNS, atomic_json, get_run, utcnow


def execute(directory):
    directory = Path(directory).resolve()
    if directory.parent != RUNS.resolve():
        raise ValueError("Run must be a direct child of GUI runs directory")
    state = get_run(directory.name)
    log_file = directory / "process.log"
    cancellation = directory / "cancel.flag"
    state.update(status="running", started=utcnow(), pid=os.getpid())
    atomic_json(directory / "run.json", state)
    outcome = "completed"
    returncode = 0
    try:
        with log_file.open("a", encoding="utf-8", buffering=1) as log:
            for index, step in enumerate(state["steps"]):
                if cancellation.exists():
                    outcome, returncode = "cancelled", 130
                    break
                state["step_index"] = index
                state["current_step"] = step["name"]
                atomic_json(directory / "run.json", state)
                log.write(f"\n===== [{index+1}/{len(state['steps'])}] {step['name']} =====\n")
                started = time.monotonic()
                with subprocess.Popen(
                    step["argv"], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    close_fds=True, env={**os.environ, "PYTHONUNBUFFERED": "1"}
                ) as process:
                    state["child_pid"] = process.pid
                    atomic_json(directory / "run.json", state)
                    while True:
                        try:
                            code = process.wait(timeout=.5)
                            break
                        except subprocess.TimeoutExpired:
                            if cancellation.exists():
                                state["status"] = "cancel_requested"
                                atomic_json(directory / "run.json", state)
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                                outcome, returncode = "cancelled", 130
                                code = returncode
                                break
                state["child_pid"] = None
                state.setdefault("results", []).append({
                    "name": step["name"], "exit_code": code,
                    "seconds": round(time.monotonic()-started, 3)})
                log.write(f"===== EXIT {code} =====\n")
                atomic_json(directory / "run.json", state)
                if outcome == "cancelled":
                    break
                if code:
                    outcome, returncode = "failed", int(code)
                    break
        if cancellation.exists() and outcome == "completed":
            outcome, returncode = "cancelled", 130
    except Exception:
        outcome, returncode = "failed", 1
        state["error"] = traceback.format_exc()
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n" + state["error"] + "\n")
    state.update(status=outcome, returncode=returncode, finished=utcnow(),
                 current_step=None, child_pid=None)
    if outcome == "completed":
        state["step_index"] = len(state["steps"])
    atomic_json(directory / "run.json", state)
    return returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.run)


if __name__ == "__main__":
    sys.exit(main())
