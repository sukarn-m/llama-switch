#!/bin/bash
# Perplexity with the ai-dock llama.cpp build (the Unsloth engine ships no
# llama-perplexity launcher). Runs on GPU0 only: the NCCL stub would crash on
# a multi-GPU split. Stop llama-switch (or unload the model) first.
#
#   ppl.sh MODEL.gguf CORPUS [extra llama-perplexity args...]
#
# Defaults: -ngl 99 -fa on -c 8192 -b 2048 -ub 512 -ctk q8_0 -ctv q5_0
# (later args win, e.g. "-ctv q8_0"). Compare runs only on the same corpus
# and -c. Corpora used so far: wikitext-2 test split (see AGENTS.md
# "Measuring perplexity"; qwen-27b ~6.5) and work/corpus.txt truncated to
# 1.5 MB with `head -c 1500000` (qwen-27b ~2.07 at -c 4096 --chunks 40).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${CTXTEST_WORK:-$here/work}"
S="${LLAMA_CPP_CUDA:-$HOME/sources/llama.cpp-cuda}"
V="$HOME/.cache/lm-studio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
[[ -s "$work/nccl-stub/libnccl.so.2" ]] || "$here/setup.sh" >/dev/null
[[ $# -ge 2 ]] || { sed -n '2,13p' "$0"; exit 1; }
model="$1"; corpus="$2"; shift 2
LD_LIBRARY_PATH="$S:$V:$work/nccl-stub:/lib64" CUDA_VISIBLE_DEVICES=0 \
    "$S/llama-perplexity" -m "$model" -f "$corpus" \
    -ngl 99 -fa on -c 8192 -b 2048 -ub 512 -ctk q8_0 -ctv q5_0 "$@" 2>&1 \
    | grep -E 'Final estimate|error|failed|^\[' | tail -5
