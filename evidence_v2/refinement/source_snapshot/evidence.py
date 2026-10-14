"""One command for tests and fresh strict benchmark evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .benchmark import run
from .gates import verify_benchmark


def run_checks(output):
    output.mkdir(parents=True,exist_ok=True)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],capture_output=True,text=True)
    (output/'verification.json').write_text(json.dumps({'exit_code':tests.returncode,'test_output':tests.stdout+tests.stderr},indent=2),encoding='utf-8')
    print(tests.stdout+tests.stderr)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    run(output)
    verify_benchmark(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('my_results'));a=p.parse_args()
    run_checks(a.output)
