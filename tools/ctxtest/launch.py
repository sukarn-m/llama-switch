#!/usr/bin/env python3
"""Start a test llama-server for a model from config/config.yaml.

The argument list is built the same way llama-switch builds it
(ModelConfig.BuildArgs in src/config.go): model fields, then
backend.common_args minus flags the model overrides, then extra_args.
On top of that you can change flags for the experiment:

  launch.py qwen-27b                             # exactly what llama-switch runs
  launch.py qwen-27b -- --spec-draft-n-max 4     # replace one flag
  launch.py qwen-27b --drop spec-draft-model,spec-draft-device \\
      -- -m /path/other.gguf --spec-type draft-mtp
  launch.py qwen-27b --print                     # show the command, don't run
  launch.py --stop                               # kill the test server

Every flag after "--" replaces the same flag (and its values) wherever it
appears in the built list. Spell flags the way config.yaml does (for
example "--override-tensor", not "-ot"), otherwise both copies are passed.

The server listens on 127.0.0.1:$CTXTEST_PORT (default 8300), logs to
work/<name>.log, and this script waits until /health answers, then prints
nvidia-smi free memory per GPU. Stop llama-switch first
(systemctl --user stop llama-switch) so nothing else loads a model.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PORT, REPO, URL, gpu_free, work_path  # noqa: E402


def expand(s):
    """Mirror of expand() in src/config.go: ~ and $VAR in each ':' part."""
    parts = []
    for p in str(s).split(":"):
        if p == "~" or p.startswith("~/"):
            p = os.path.expanduser(p)
        parts.append(os.path.expandvars(p))
    return ":".join(parts)


def flag_name(tok):
    if not tok.startswith("-") or len(tok) == 1:
        return None
    try:
        float(tok)
        return None
    except ValueError:
        pass
    return tok.split("=", 1)[0]


def drop_flags(args, names):
    """Remove each flag in names together with its values (tokens up to the next flag)."""
    out, skipping = [], False
    for tok in args:
        n = flag_name(tok)
        if n is not None:
            skipping = n in names
        if not skipping:
            out.append(tok)
    return out


def build(cfg, mid):
    be = cfg.get("backend", {})
    m = next((x for x in cfg["models"] if x.get("id") == mid
              or str(x.get("name", "")).lower() == mid.lower()), None)
    if m is None:
        sys.exit(f"model {mid!r} not in config; ids: " + ", ".join(x["id"] for x in cfg["models"]))
    if m.get("args"):
        sys.exit(f"{mid} uses a raw args list; launch it by hand")
    a = ["-m", expand(m["path"])]
    if m.get("mmproj"):
        a += ["--mmproj", expand(m["mmproj"])]
    if m.get("name"):
        a += ["--alias", m["name"]]
    a += ["-c", str(m.get("context_size", 0))]
    if m.get("parallel"):
        a += ["--parallel", str(m["parallel"])]
    if m.get("devices"):
        a += ["--device", ",".join(m["devices"])]
    if m.get("tensor_split"):
        a += ["--tensor-split", str(m["tensor_split"])]
    cc = m.get("ctx_checkpoints") or be.get("ctx_checkpoints")
    if cc:
        a += ["--ctx-checkpoints", str(cc)]
    cr = m["cache_ram"] if "cache_ram" in m else be.get("cache_ram")
    if cr is not None and (cr != 0 or "cache_ram" in m):
        a += ["--cache-ram", str(cr)]
    a += ["--host", "127.0.0.1", "--port", str(PORT)]
    extra = [str(x) for x in m.get("extra_args", [])]
    common = [str(x) for x in be.get("common_args", [])]
    a += drop_flags(common, {flag_name(t) for t in extra if flag_name(t)})
    a += extra
    binary = expand(m.get("binary") or be["binary"])
    env = dict(os.environ)
    env.update({k: expand(v) for k, v in (be.get("env") or {}).items()})
    env.update({k: expand(v) for k, v in (m.get("env") or {}).items()})
    return binary, a, env, expand(be.get("workdir", "~"))


def stop():
    out = subprocess.run(["pgrep", "-f", f"llama-server.*--port {PORT}( |$)"],
                         capture_output=True, text=True).stdout.split()
    for pid in out:
        os.kill(int(pid), signal.SIGTERM)
    for _ in range(30):
        if not subprocess.run(["pgrep", "-f", f"llama-server.*--port {PORT}( |$)"],
                              capture_output=True).stdout:
            return
        time.sleep(1)


def main():
    argv = sys.argv[1:]
    overrides = []
    if "--" in argv:
        i = argv.index("--")
        argv, overrides = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?")
    p.add_argument("--config", default=os.path.join(REPO, "config", "config.yaml"))
    p.add_argument("--name", help="log name (default: model id)")
    p.add_argument("--drop", action="append", default=[], metavar="FLAG",
                   help="remove FLAG (without leading dashes, comma-separated or repeated) and its values")
    p.add_argument("--print", action="store_true", help="print the command and exit")
    p.add_argument("--stop", action="store_true", help="stop the test server and exit")
    p.add_argument("--timeout", type=int, default=300)
    o = p.parse_args(argv)
    if o.stop:
        stop()
        return
    if not o.model:
        p.error("model id required")

    cfg = yaml.safe_load(open(o.config))
    binary, args, env, wd = build(cfg, o.model)
    if o.drop:
        names = {n if n.startswith("-") else ("-" if len(n) == 1 else "--") + n
                 for d in o.drop for n in d.split(",") if n}
        args = drop_flags(args, names)
    if overrides:
        args = drop_flags(args, {flag_name(t) for t in overrides if flag_name(t)}) + overrides
    if o.print:
        print(binary, " ".join(f"'{x}'" if any(c in x for c in " |()[]\\*") else x for x in args))
        return

    stop()
    log = work_path(f"{o.name or o.model}.log")
    with open(log, "w") as lf:
        lf.write(" ".join([binary] + args) + "\n\n")
        lf.flush()
        proc = subprocess.Popen([binary] + args, env=env, cwd=wd, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < o.timeout:
        if proc.poll() is not None:
            print(f"DIED (exit {proc.returncode}); last lines of {log}:")
            lines = open(log, errors="replace").read().splitlines()
            print("\n".join(l for l in lines[-40:] if " E " in l or "error" in l.lower() or "memory" in l) or "\n".join(lines[-15:]))
            sys.exit(1)
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            free = gpu_free()
            print(f"UP after {time.time() - t0:.0f}s on {URL}; idle free: "
                  + "  ".join(f"GPU{i} {v} MiB" for i, v in enumerate(free)))
            return
        except Exception:
            time.sleep(2)
    print(f"TIMEOUT after {o.timeout}s; see {log}")
    sys.exit(1)


if __name__ == "__main__":
    main()
