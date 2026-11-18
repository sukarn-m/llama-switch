#!/usr/bin/env python3
"""Long-context stress and recall test against the running test server.

  needle.py TARGET_TOKENS [extra-json] [--image]

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
from _common import MinFree, chat, work_path  # noqa: E402

FACTS = [
    ("// NOTE: The registration number of the sale deed is RJ-4471-B.", "RJ-4471-B"),
    ("// NOTE: The purchaser's name is Kamla Devi Meena.", "Kamla Devi Meena"),
    ("// NOTE: The agreed sale consideration is Rs. 38,65,000.", "38,65,000"),
]


def main():
    args = [a for a in sys.argv[1:] if a != "--image"]
    image = "--image" in sys.argv
    if not args:
        sys.exit(__doc__)
    target = int(args[0])
    extra = json.loads(args[1]) if len(args) > 1 and args[1] else {}

    text = open(work_path("corpus.txt"), encoding="utf-8", errors="ignore").read()
    body_text = text[: int(target * 3.25)]
    if len(body_text) < target * 3.25 * 0.95:
        sys.exit("corpus.txt is too short for that target; rebuild it larger with setup.sh")
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


if __name__ == "__main__":
    main()
