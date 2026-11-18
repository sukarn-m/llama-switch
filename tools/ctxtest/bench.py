#!/usr/bin/env python3
"""Short-prompt generation speed of the running test server.

  bench.py [runs] [extra-json]

Sends three prompts (code, legal English, Hindi translation) with thinking
off and max_tokens 600, `runs` times each (default 3), after one warm-up.
Prints the mean generation t/s, each sample, and the speculative draft
acceptance rate. Single samples vary by +-8 t/s with drafting on, so compare
means over at least 2-3 runs. extra-json is merged into every request body,
e.g. '{"temperature":0.7}'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import chat  # noqa: E402

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the difference between a sale deed and an agreement to sell under Indian law, in about 300 words.",
    "Translate to Hindi: The vendor hereby conveys the property free from all encumbrances to the purchaser.",
]


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    def call(p):
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 600,
                "chat_template_kwargs": {"enable_thinking": False}}
        body.update(extra)
        return chat(body, timeout=900)

    call("hi")
    tps, drafted, accepted = [], 0, 0
    for _ in range(runs):
        for p in PROMPTS:
            t = call(p)["timings"]
            tps.append(t["predicted_per_second"])
            drafted += t.get("draft_n", 0)
            accepted += t.get("draft_n_accepted", 0)
    acc = f"{accepted / drafted:.2f}" if drafted else "n/a"
    print(f"gen avg {sum(tps) / len(tps):.1f} t/s  samples {[round(x, 1) for x in tps]}  draft accept {acc}")


if __name__ == "__main__":
    main()
