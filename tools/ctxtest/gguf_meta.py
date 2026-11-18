#!/usr/bin/env python3
"""Print a GGUF's metadata (minus the vocab arrays) and tensor-type counts.

  gguf_meta.py FILE.gguf

Uses the gguf-py that ships with the Unsloth llama.cpp tree when present,
because the pip `gguf` package lags behind new quant types (it failed on
Bonsai's Q2_0, type 42). Useful fields: <arch>.block_count,
<arch>.nextn_predict_layers (built-in MTP head), <arch>.context_length,
attention head counts, and the per-type tensor counts.
"""
import collections
import os
import sys

unsloth = os.path.expanduser("~/.unsloth/llama.cpp/gguf-py")
if os.path.isdir(unsloth):
    sys.path.insert(0, unsloth)
from gguf import GGUFReader  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
        continue
    if k == "tokenizer.chat_template":
        print(k, f"(template, {len(bytes(f.parts[-1]))} bytes)")
        continue
    try:
        v = f.contents()
    except Exception:
        v = "?"
    print(k, "=", str(v)[:200])
print("tensor types:", dict(collections.Counter(t.tensor_type.name for t in r.tensors)), "total", len(r.tensors))
print("block 0:", [(t.name, t.tensor_type.name) for t in r.tensors if t.name.startswith("blk.0.")])
