"""Shared helpers for the ctxtest harness (imported by the other scripts)."""
import json
import os
import subprocess
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("CTXTEST_WORK", os.path.join(HERE, "work"))
# Outside llama-switch's backend range (8201-8299) so a running service can't
# collide with the test server.
PORT = int(os.environ.get("CTXTEST_PORT", "8300"))
URL = f"http://127.0.0.1:{PORT}"


def chat(body, timeout=3600):
    """POST /v1/chat/completions to the test server and return the JSON."""
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gpu_free():
    """nvidia-smi memory.free per GPU, in MiB (the figure the headroom rule uses)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(x) for x in out.split()]


class MinFree:
    """Context manager that samples gpu_free() every 0.5 s and keeps the minimum."""

    def __enter__(self):
        self.min = gpu_free()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.5):
            try:
                self.min = [min(a, b) for a, b in zip(self.min, gpu_free())]
            except Exception:
                pass

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def __str__(self):
        return "  ".join(f"GPU{i} {v} MiB" for i, v in enumerate(self.min))


def work_path(*parts):
    os.makedirs(WORK, exist_ok=True)
    return os.path.join(WORK, *parts)


def now():
    return time.time()


# ── Saved results ─────────────────────────────────────────────
# Every test writes one JSON file per run to results/<model-id>/ (gitignored
# like work/, but kept: never clear it). The run record that launch.py leaves in work/run.json
# says what was running: model, binary, engine version, exact args and any
# overrides, so a result can always be traced to its setup.

RESULTS = os.environ.get("CTXTEST_RESULTS", os.path.join(HERE, "results"))
SCHEMA = 1


def run_record():
    """The launch record of the running test server (work/run.json)."""
    try:
        with open(work_path("run.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit("no work/run.json: start the server with launch.py first")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def save_result(test, params, metrics, data=None):
    """Write results/<model>/<timestamp>-<test>[-<variant>].json and return its path.

    params: what the test was asked to do; metrics: the numbers it measured;
    data: bulky extras (quiz answers, OCR text). The run record is copied in.
    """
    run = run_record()
    ts = time.strftime("%Y%m%d-%H%M%S")
    doc = {
        "schema": SCHEMA,
        "test": test,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": run["model"],
        "variant": run.get("variant"),
        "production": run.get("production", False),
        "run": run,
        "llama_switch_commit": _git("rev-parse", "--short", "HEAD"),
        "llama_switch_dirty": bool(_git("status", "--porcelain", "--", "config", "src")),
        "gpus": gpus(),
        "params": params,
        "metrics": metrics,
    }
    if data is not None:
        doc["data"] = data
    d = os.path.join(RESULTS, run["model"])
    os.makedirs(d, exist_ok=True)
    name = f"{ts}-{test}" + (f"-{run['variant']}" if run.get("variant") else "") + ".json"
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return path


def props():
    """GET /props of the test server (n_ctx etc.)."""
    with urllib.request.urlopen(URL + "/props", timeout=30) as r:
        return json.load(r)


def tokenize(text):
    req = urllib.request.Request(URL + "/tokenize", json.dumps({"content": text}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(json.load(r)["tokens"])
