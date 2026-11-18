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
