# ctxtest: model fit, speed and quality harness

Scripts for trying a new model, quant, drafter or layout before it goes into
`config/config.yaml`. They run one llama-server outside llama-switch, built
from the same config, so what you measure is what the service would run.

Generated files (logs, corpus, quiz answers, the NCCL stub) go to `work/`,
which is gitignored. Override with `CTXTEST_WORK`. The test server listens on
`127.0.0.1:8300` (`CTXTEST_PORT`), outside llama-switch's backend port range.

## Before you start

```bash
systemctl --user stop llama-switch   # nothing else may load a model meanwhile
tools/ctxtest/setup.sh               # once: corpus, test image, NCCL stub
```

Restart the service when done: `systemctl --user start llama-switch`.

## Scripts

| Script | What it does |
|---|---|
| `launch.py MODEL_ID [--drop a,b] [-- flags…]` | Builds the llama-server command for a config model the way `BuildArgs` does, applies your changes, starts it, waits for `/health`, prints idle free VRAM per GPU. `--print` shows the command only; `--stop` kills the test server. Log: `work/<id>.log`. |
| `bench.py [runs] [json]` | Short-prompt generation speed (3 prompts × runs, thinking off) and draft acceptance. |
| `needle.py TOKENS [json] [--image]` | Long-context stress + recall: 3 facts at 10/50/90% depth, optional image, minimum free VRAM per GPU during the request. |
| `quiz.py LABEL [json]` | Six Indian-law / Hindi-deed questions with thinking on; answers and an answer key go to `work/quiz-LABEL.txt` for grading by hand. |
| `ppl.sh MODEL CORPUS [args]` | Perplexity with the ai-dock build (GPU0 only). |
| `gguf_meta.py FILE` | GGUF metadata and tensor types (knows newer quant types via Unsloth's gguf-py). |

`[json]` is merged into each request body, e.g. `'{"temperature":0.7}'`.

## Typical session

```bash
cd tools/ctxtest
./launch.py qwen-27b                                    # baseline, as in production
./bench.py 3
./needle.py 250000 '' --image                           # must keep >= 600 MiB free on GPU0

# Same layout with another GGUF and its built-in MTP head instead of DFlash:
./launch.py qwen-27b --name candidate --drop spec-draft-model,spec-draft-device -- \
    -m /path/to/candidate.gguf --spec-type draft-mtp --spec-draft-n-max 3
./bench.py 3 && ./quiz.py candidate
./launch.py --stop
```

Flags after `--` replace the same flag wherever it appears in the built
command (common args included). Spell them as the config does
(`--override-tensor`, not `-ot`), or both copies are passed.

## Reading the numbers

- **Headroom:** use the `MIN FREE` line from `needle.py` at ~250k tokens with
  `--image`, not the idle figure. On qwen-27b, free memory on GPU0 dropped about
  1.5 GB from idle to the peak of that test, so a layout that idles under ~2 GB
  free on GPU0 is unlikely to pass.
- **Speed:** with speculative decoding, single samples swing by ±8 t/s;
  compare means of at least 2–3 runs. A drafter with acceptance under
  ~0.3 is usually slower than none: DFlash2 drafters trained on a base
  model do not transfer to finetunes or rotated-weight quants.
- **Quality:** `quiz.py` is one sample per question. Disqualify on confident
  fabrication (invented section text or provisos, changed amounts), not on
  a single wrong section number. Use `ppl.sh` for quant or KV-cache changes
  of the *same* model; perplexity across different finetunes is not a
  quality ranking.
