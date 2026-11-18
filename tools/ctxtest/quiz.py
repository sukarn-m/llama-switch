#!/usr/bin/env python3
"""Indian-law + Hindi sale-deed spot check for the running test server.

  quiz.py LABEL [extra-json]

Asks six questions with thinking ON (max_tokens 16000) and writes each
answer, with token counts, speed and finish reason, to work/quiz-LABEL.txt.
Grade by hand against KEY below. One sample per question at the model's
default temperature, so treat a difference of one question between models
as noise; look for confident fabrication (invented section text, invented
provisos, changed amounts), which is what disqualifies a model.

extra-json is merged into each request body, e.g.
'{"temperature":0.7}' or '{"chat_template_kwargs":{"reasoning_effort":"medium"}}'.

Results so far (Nov 18 2026, one sample each):
  unsloth Qwen3.8-27B UD-Q4_K_M: s53A, lim, deed right; s54 gave the s.5
    "transfer" definition; reg23 four months but cited s.27; s49 invented
    exceptions.
  DavidAU Qwen3.8-27B TURBO Heretic MTP Q4_K_M (temp 0.7): s54, s49
    (real proviso), lim, deed right; s53A mostly right; reg23 cited s.17.
  prism-ml Ternary-Bonsai-2-27B Q2_0: fabricated every statute answer and
    read Rs 18,50,000 as Rs 1,85,00,000. Rejected.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import chat, work_path  # noqa: E402

DEED = """यह विक्रय पत्र आज दिनांक 14 मार्च 2025 को श्री रामेश्वर लाल पुत्र श्री गोपाल राम, निवासी ग्राम बस्सी, तहसील बस्सी, जिला जयपुर (जिसे आगे विक्रेता कहा गया है) तथा श्रीमती कमला देवी पत्नी श्री महेश कुमार मीणा, निवासी 12, शांति नगर, जयपुर (जिसे आगे क्रेता कहा गया है) के मध्य निष्पादित किया गया। विक्रेता ग्राम बस्सी की खसरा संख्या 482/2 रकबा 0.35 हेक्टेयर कृषि भूमि का एकमात्र स्वामी एवं काबिज है, जो उसे अपने पिता से उत्तराधिकार में प्राप्त हुई। विक्रेता ने उक्त भूमि क्रेता को कुल प्रतिफल राशि रुपये 18,50,000/- (अठारह लाख पचास हजार रुपये) में विक्रय की, जिसमें से रुपये 2,00,000/- बयाना दिनांक 10 जनवरी 2025 को नकद प्राप्त हुए तथा शेष राशि रुपये 16,50,000/- आज चेक संख्या 004417, भारतीय स्टेट बैंक, शाखा बस्सी द्वारा प्राप्त की। भूमि पर कोई बंधक, ऋण या भार नहीं है। मालियत अनुसार डीएलसी दर रुपये 21,00,000/- है जिस पर स्टाम्प शुल्क अदा किया गया है।"""

QUESTIONS = [
    ("s54", "Under Indian law, what does section 54 of the Transfer of Property Act, 1882 provide? "
            "Mention when a registered instrument is required."),
    ("s53A", "What is the doctrine embodied in section 53A of the Transfer of Property Act, 1882, "
             "and what are its essential conditions?"),
    ("reg23", "Under the Registration Act, 1908, within what time must a document (other than a will) "
              "be presented for registration after execution, and which section says so?"),
    ("lim", "What is the limitation period for a suit for specific performance of a contract in India, "
            "which Article of the Limitation Act, 1963 governs it, and from when does time begin to run?"),
    ("s49", "What is the effect of non-registration of a document that is compulsorily registrable under "
            "section 17 of the Registration Act, 1908? Cite the section and its proviso."),
    ("deed", "नीचे दिए गए विक्रय पत्र के अंश से निम्न विवरण अंग्रेज़ी में निकालिए: (1) विक्रेता का नाम व पिता का नाम, "
             "(2) क्रेता का नाम, (3) खसरा संख्या व रकबा, (4) कुल प्रतिफल, बयाना व शेष राशि तथा भुगतान का तरीका, "
             "(5) स्टाम्प शुल्क किस राशि पर अदा किया गया और क्यों, (6) भूमि पर स्वामित्व का स्रोत।\n\n" + DEED),
]

# What a correct answer must contain.
KEY = {
    "s54": "Defines 'sale': transfer of ownership for a price paid, promised or part-paid/part-promised. "
           "Tangible immovable property of value Rs 100 or more (or a reversion/intangible thing): only by "
           "registered instrument; below Rs 100: registered instrument or delivery of property. A contract "
           "for sale creates no interest or charge in the property.",
    "s53A": "Part performance. Written contract signed by/on behalf of transferor, terms ascertainable with "
            "reasonable certainty; transferee has taken possession (or, if already in possession, continues) "
            "in part performance; has done some act in furtherance; has performed or is willing to perform "
            "his part. Transferor is debarred from enforcing rights other than those under the contract. "
            "A shield, not a sword; does not affect a transferee for consideration without notice.",
    "reg23": "Section 23: within four months from execution (s.24 for several executants; s.25 lets the "
             "Registrar admit within a further four months on a fine up to ten times the fee; s.26 for "
             "documents executed outside India).",
    "lim": "Article 54, Limitation Act 1963 (Schedule, First Division): three years from the date fixed for "
           "performance, or if no date is fixed, from when the plaintiff has notice that performance is refused.",
    "s49": "Section 49: an unregistered compulsorily registrable document does not affect the immovable "
           "property, does not confer a power to adopt, and is not received as evidence of a transaction "
           "affecting the property. Proviso: it may be received as evidence of a contract in a suit for "
           "specific performance (Specific Relief Act, Ch. II), as evidence of part performance under s.53A "
           "TPA, or of any collateral transaction not required to be effected by registered instrument.",
    "deed": "Seller Rameshwar Lal s/o Gopal Ram (village/tehsil Bassi, Jaipur); buyer Smt. Kamla Devi w/o "
            "Mahesh Kumar Meena; khasra 482/2, 0.35 ha agricultural; consideration Rs 18,50,000, earnest "
            "Rs 2,00,000 cash on 10 Jan 2025, balance Rs 16,50,000 by SBI Bassi cheque no. 004417 on the "
            "deed date; stamp duty on the DLC value Rs 21,00,000 because it exceeds the consideration; "
            "title by inheritance from father; no encumbrance.",
}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    label = sys.argv[1]
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    path = work_path(f"quiz-{label}.txt")
    with open(path, "w") as out:
        for k, q in QUESTIONS:
            body = {"messages": [{"role": "user", "content": q}], "max_tokens": 16000}
            body.update(extra)
            t0 = time.time()
            j = chat(body)
            m, t = j["choices"][0]["message"], j["timings"]
            out.write(f"===== {k}  gen_tokens={t['predicted_n']} reasoning_chars={len(m.get('reasoning_content') or '')} "
                      f"{t['predicted_per_second']:.1f}t/s {time.time() - t0:.0f}s finish={j['choices'][0]['finish_reason']}\n"
                      f"KEY: {KEY[k]}\n\n{m.get('content')}\n\n")
            out.flush()
            print(k, t["predicted_n"], f"{time.time() - t0:.0f}s", flush=True)
    print("answers in", path)


if __name__ == "__main__":
    main()
