#!/usr/bin/env python3
"""Long-context stress and recall test against the running test server.

  needle.py TARGET_TOKENS [extra-json] [--image] [--tag NAME]
  needle.py --fill FRACTION [extra-json] [--image] [--tag NAME]

--fill sizes the prompt to FRACTION of the server's context (from /props),
measured with the server's own tokenizer, leaving room for the image and the
answer; use --fill 0.95 for the worst-case stress test. Results are saved to
results/<model>/<time>-needle.json; --tag names the run in the summary
(the suite uses "pp30k" for a ~30k-token prompt-speed run and "stress").

Builds a prompt of roughly TARGET_TOKENS tokens from work/corpus.txt (Go
source, ~3.25 chars/token), hides three facts at 10%, 50% and 90% depth,
and asks the model to quote them. With --image, work/test.png is attached
too, so the vision encoder runs in the same request (the worst case for
VRAM). While the request runs, nvidia-smi free memory is sampled every
0.5 s; the minimum per GPU is the number to hold against the headroom rule
(>= 600 MiB on GPU0, aim for ~1 GB).

Run setup.sh once first to create the corpus and the image. Thinking is
off unless extra-json turns it on. For Muse Glimmer pass
'{"chat_template_kwargs":{"reasoning_strength":"low"}}'.
"""
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import MinFree, chat, props, save_result, tokenize, work_path  # noqa: E402

FACTS = [
    ("// NOTE: The registration number of the sale deed is RJ-4471-B.", "RJ-4471-B"),
    ("// NOTE: The purchaser's name is Kamla Devi Meena.", "Kamla Devi Meena"),
    ("// NOTE: The agreed sale consideration is Rs. 38,65,000.", "38,65,000"),
]


def take(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return None


def main():
    argv = sys.argv[1:]
    image = "--image" in argv
    argv = [a for a in argv if a != "--image"]
    tag = take(argv, "--tag")
    fill = take(argv, "--fill")
    if not argv and fill is None:
        sys.exit(__doc__)
    extra = json.loads(argv[-1]) if (argv and (fill is not None or len(argv) > 1) and argv[-1]) else {}

    text = open(work_path("corpus.txt"), encoding="utf-8", errors="ignore").read()
    if fill is not None:
        n_ctx = props()["default_generation_settings"]["n_ctx"]
        # room for the question, the image (up to ~4k tokens) and max_tokens 2500
        target = int(n_ctx * float(fill)) - (2500 + 300 + (4500 if image else 0))
        sample = text[:400000]
        cpt = len(sample) / tokenize(sample)
    else:
        target = int(argv[0])
        cpt = 3.25
    body_text = text[: int(target * cpt)]
    if len(body_text) < target * cpt * 0.95:
        sys.exit("corpus.txt is too short for that target; rebuild it larger with setup.sh")
    if fill is not None:
        # tokenizer ratios drift along the corpus: trim until the real count fits
        while (n := tokenize(body_text)) > target:
            body_text = body_text[: int(len(body_text) * target / n * 0.995)]
    # Insert deepest first so earlier offsets stay valid.
    for frac, (line, _) in sorted(zip([0.1, 0.5, 0.9], FACTS), key=lambda x: -x[0]):
        pos = body_text.find("\n", int(len(body_text) * frac))
        body_text = body_text[: pos + 1] + line + "\n" + body_text[pos + 1:]
    question = ("\n\nThe code above contains three hidden comments starting with '// NOTE:'. "
                "Quote all three exactly, then list: registration number, purchaser's name, sale consideration.")
    content = [{"type": "text", "text": body_text + question}]
    if image:
        b64 = base64.b64encode(open(work_path("test.png"), "rb").read()).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 2500,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)

    t0 = time.time()
    with MinFree() as mf:
        r = chat(body)
    t = r["timings"]
    c = r["choices"][0]["message"].get("content") or ""
    print("prompt tokens", t["prompt_n"] + t.get("cache_n", 0))
    print(f"pp {t['prompt_per_second']:.0f} t/s  gen {t['predicted_per_second']:.1f} t/s  wall {time.time() - t0:.0f}s")
    print("HITS (10%/50%/90%):", [key in c for _, key in FACTS])
    print("MIN FREE during request:", mf)
    print("CONTENT tail:", c[-500:].replace("\n", " "))
    hits = [key in c for _, key in FACTS]
    path = save_result("needle",
                       {"tag": tag, "target_tokens": target, "fill": float(fill) if fill else None,
                        "image": image, "extra": extra},
                       {"prompt_tokens": t["prompt_n"] + t.get("cache_n", 0),
                        "pp_tps": round(t["prompt_per_second"], 1), "gen_tps": round(t["predicted_per_second"], 2),
                        "wall_s": round(time.time() - t0), "hits": hits, "recall": f"{sum(hits)}/{len(hits)}",
                        "min_free_mib": mf.min, "finish": r["choices"][0]["finish_reason"]},
                       {"content_tail": c[-1500:]})
    print("saved", path)


if __name__ == "__main__":
    main()
