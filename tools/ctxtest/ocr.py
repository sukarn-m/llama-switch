#!/usr/bin/env python3
"""Hindi OCR test: transcribe a rendered sale-deed page and score it.

  ocr.py [extra-json]

Renders the quiz's deed paragraph (quiz.DEED) onto an A4 page at 150 dpi
with Noto Serif Devanagari (work/ocr-deed.png, created once; its sha256 is
recorded), asks the running server to transcribe it with thinking off, and
scores the answer against the source text:

  cer         character error rate after Unicode NFC, with whitespace,
              zero-width joiners and Markdown/HTML markup removed
              (0 = perfect; a dropped conjunct counts as an error)
  fields      how many of the key facts (names, khasra, area, amounts,
              cheque number, dates) appear verbatim, out of len(FIELDS)

Also records image prompt tokens, prompt time, wall time and the minimum
free VRAM per GPU during the request. Saves results/<model>/<time>-ocr.json.
For a reasoning model whose OCR text lands in reasoning_content (Chandra),
that field is used when content is empty.
"""
import base64
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import MinFree, chat, save_result, work_path  # noqa: E402
from quiz import DEED  # noqa: E402

TITLE = "विक्रय पत्र"
FONT = "/usr/share/fonts/google-noto-vf/NotoSerifDevanagari[wght].ttf"
PROMPT = ("Transcribe all the text in this image exactly as written, in the original Hindi script. "
          "Keep the numbers and punctuation as they are. Output only the transcription.")
FIELDS = ["रामेश्वर लाल", "गोपाल राम", "कमला देवी", "महेश कुमार मीणा", "बस्सी", "482/2", "0.35",
          "18,50,000", "2,00,000", "16,50,000", "004417", "21,00,000", "14 मार्च 2025", "10 जनवरी 2025"]


def render(path):
    from PIL import Image, ImageDraw, ImageFont
    W, H, margin = 1240, 1754, 110
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title = ImageFont.truetype(FONT, 44)
    body = ImageFont.truetype(FONT, 30)
    d.text((W // 2, margin), TITLE, font=title, fill="black", anchor="mt")
    y, line = margin + 110, ""
    for word in DEED.split():
        trial = (line + " " + word).strip()
        if d.textlength(trial, font=body) > W - 2 * margin:
            d.text((margin, y), line, font=body, fill="black")
            y, line = y + 52, word
        else:
            line = trial
    d.text((margin, y), line, font=body, fill="black")
    im.save(path)


def norm(s):
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"<[^>]+>", " ", s)                 # HTML tags (Chandra, tables)
    s = re.sub("[*#|_`>~\u200b\u200c\u200d]", "", s)  # Markdown and zero-width chars
    return re.sub(r"\s+", "", s)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
    img = work_path("ocr-deed.png")
    if not os.path.exists(img):
        render(img)
    raw = open(img, "rb").read()
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}},
               {"type": "text", "text": PROMPT}]
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 3000, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)
    t0 = time.time()
    with MinFree() as mf:
        r = chat(body, timeout=1800)
    wall = time.time() - t0
    m, t = r["choices"][0]["message"], r["timings"]
    text = m.get("content") or m.get("reasoning_content") or ""
    hyp = norm(text)
    ref = norm(TITLE + DEED)  # the page has the title line, then the deed paragraph
    cer = edit_distance(hyp, ref) / len(ref)
    found = [f for f in FIELDS if norm(f) in norm(text)]
    print(f"CER {cer:.3f}  fields {len(found)}/{len(FIELDS)}  image+prompt tokens {t['prompt_n']}  "
          f"pp {t['prompt_ms'] / 1000:.1f}s  wall {wall:.1f}s  min free {mf}")
    missing = [f for f in FIELDS if f not in found]
    if missing:
        print("missing:", ", ".join(missing))
    path = save_result("ocr",
                       {"image": os.path.basename(img), "image_sha256": hashlib.sha256(raw).hexdigest(),
                        "font": os.path.basename(FONT), "prompt": PROMPT, "extra": extra},
                       {"cer": round(cer, 4), "fields_found": len(found), "fields_total": len(FIELDS),
                        "prompt_tokens": t["prompt_n"], "prompt_s": round(t["prompt_ms"] / 1000, 2),
                        "gen_tps": round(t["predicted_per_second"], 2), "wall_s": round(wall, 1),
                        "min_free_mib": mf.min, "finish": r["choices"][0]["finish_reason"]},
                       {"text": text, "missing_fields": missing})
    print("saved", path)


if __name__ == "__main__":
    main()
