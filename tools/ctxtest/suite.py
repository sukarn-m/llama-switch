#!/usr/bin/env python3
"""Run the standard test suite (suite.yaml) on one or more models.

  suite.py MODEL [MODEL ...]      # or: suite.py all
  suite.py all --only ocr,bench   # a subset of the tests

For each model: launch it exactly as config.yaml says (a production run),
run its tests, stop it. Each test saves its own result file; a failed test is
logged and the suite moves on. llama-switch is stopped for the duration and
restarted at the end if it was running. Finish with report.py; quiz results
still need grading (grade.py).
"""
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import work_path  # noqa: E402

COMMANDS = {
    "bench": ["./bench.py", "3"],
    "pp30k": ["./needle.py", "26000", "--tag", "pp30k"],
    "stress": ["./needle.py", "--fill", "0.95", "--image", "--tag", "stress"],
    "ocr": ["./ocr.py"],
    "quiz": ["./quiz.py", "--samples", "2"],
}


def service(action):
    return subprocess.run(["systemctl", "--user", action, "llama-switch"], capture_output=True, text=True)


def run(cmd, log):
    print("  $", " ".join(cmd), flush=True)
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(p.stdout)
    log.flush()
    tail = [line for line in p.stdout.splitlines() if line.strip()][-3:]
    for line in tail:
        print("    " + line[:200], flush=True)
    return p.returncode == 0


def main():
    argv = sys.argv[1:]
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = set(argv[i + 1].split(","))
        del argv[i:i + 2]
    suite = yaml.safe_load(open(os.path.join(HERE, "suite.yaml")))
    models = list(suite["models"]) if argv == ["all"] else argv
    if not models:
        sys.exit(__doc__)
    was_active = service("is-active").stdout.strip() == "active"
    if was_active:
        service("stop")
        time.sleep(3)
    log = open(work_path(f"suite-{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
    failed = []
    try:
        for m in models:
            spec = suite["models"].get(m) or {}
            tests = [t for t in spec.get("tests", suite["defaults"]["tests"]) if not only or t in only]
            print(f"== {m}: {', '.join(tests)}", flush=True)
            if not run(["./launch.py", m, "--timeout", "600"], log):
                failed.append(f"{m}: launch")
                continue
            for t in tests:
                extra = (spec.get("extra") or {}).get(t)
                cmd = COMMANDS[t] + ([json.dumps(extra)] if extra else [])
                if t == "pp30k" and not extra:
                    cmd = cmd + [""]  # needle.py takes extra-json positionally after the target
                if not run(cmd, log):
                    failed.append(f"{m}: {t}")
            run(["./launch.py", "--stop"], log)
    finally:
        run(["./launch.py", "--stop"], log)
        if was_active:
            service("start")
        log.close()
    print("FAILED:", ", ".join(failed) if failed else "none")
    print("log:", log.name)


if __name__ == "__main__":
    main()
