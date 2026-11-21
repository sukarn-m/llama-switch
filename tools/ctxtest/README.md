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
| `launch.py MODEL_ID [--drop a,b] [--binary PATH] [-- flags…]` | Builds the llama-server command for a config model the way `BuildArgs` does, applies your changes, starts it, waits for `/health`, prints idle free VRAM per GPU, and writes the run record `work/run.json` (model, binary, engine version, exact args, overrides). `--print` shows the command only; `--stop` kills the test server. Log: `work/<name>.log`. |
| `bench.py [runs] [json]` | Short-prompt generation speed (3 prompts × runs, thinking off) and draft acceptance. |
| `needle.py TOKENS [json] [--image] [--tag T]` | Long-context stress + recall: 3 facts at 10/50/90% depth, optional image, minimum free VRAM per GPU during the request. `--fill 0.95` instead of TOKENS sizes the prompt to 95% of the server's context with its own tokenizer. |
| `ocr.py [json]` | Transcribes a rendered Hindi sale-deed page (`work/ocr-deed.png`); character error rate and key fields found against the source text. |
| `quiz.py [--samples N] [json]` | Six Indian-law / Hindi-deed questions with thinking on, N samples each (default 2); answers saved with the key for grading. |
| `grade.py FILE [--set q/n=score[f]] [--note q/n=text]` | Shows a quiz result next to the key, or records grades (2/1/0, `f` = fabricated). |
| `suite.py MODEL… \| all [--only t1,t2]` | Runs the standard tests from `suite.yaml` on each model (stops and restarts llama-switch around it). |
| `report.py [--all]` | Rebuilds `results/SUMMARY.md` from the result files. |
| `ppl.sh MODEL CORPUS [args]` | Perplexity with the ai-dock build (GPU0 only). |
| `gguf_meta.py FILE` | GGUF metadata and tensor types (knows newer quant types via Unsloth's gguf-py). |

`[json]` is merged into each request body, e.g. `'{"temperature":0.7}'`.

## Saved results (`results/`, gitignored, kept)

`bench.py`, `needle.py`, `ocr.py` and `quiz.py` each write one JSON file per
run to `results/<model-id>/<YYYYMMDD-HHMMSS>-<test>[-<variant>].json`. The
repository is public, so this directory is gitignored and stays on this
machine. Unlike `work/` it is permanent: do not clear it, and back it up with
the machine, so results can be compared across engines and months. Every file has the same top level:

- `schema` (1), `test`, `timestamp`, `model`;
- `variant` and `production`: a run is production only when `launch.py`
  started the model exactly as `config.yaml` says (no `--drop`, `--binary` or
  flag overrides); otherwise `variant` is the `--name` given (or `custom`);
- `run`: the launch record (binary, engine version string, full args, GGUF
  file and size, mmproj, context, idle free VRAM, load time);
- `llama_switch_commit` / `llama_switch_dirty`, `gpus`;
- `params` (what the test was asked to do), `metrics` (what it measured) and
  `data` (bulky output: quiz answers, OCR text).

Quiz files start with an empty `grades` list. Grade every answer with
`grade.py` (2 right, 1 partly right, 0 wrong, plus a fabrication flag for
invented section or proviso text, case law or changed facts); the totals then
appear in `metrics` and in the summary. The question set is versioned
(`QUIZ_VERSION` in `quiz.py`): bump it when a question or the key changes, and
compare scores only within a version.

`report.py` rebuilds `results/SUMMARY.md`: one row per model from its latest
production run of each test, then each model's quiz notes and links to every
run. Do not edit `SUMMARY.md` by hand.

To test a new model: add it to `config.yaml` and `suite.yaml`, run
`./suite.py <id>`, grade the quiz, run `./report.py`.

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
