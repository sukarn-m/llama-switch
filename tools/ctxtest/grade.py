#!/usr/bin/env python3
"""Grade a saved quiz result.

  grade.py FILE                         # show each answer next to the key
  grade.py FILE --by NAME \\
      --set 's54/1=2' --set 's53A/2=0f' --note 's53A/2=invented oral-contract rule'

Scores: 2 = right, 1 = partly right (core point right, a detail wrong or
missing), 0 = wrong. Append "f" to flag fabrication: invented section or
proviso text, invented case law or amendments, changed amounts or names.
A wrong section number alone is a 1 or 0, not a fabrication.

Once every answer has a score, metrics.graded becomes true and the totals
(score, max, fabrications, per-question scores) are written into metrics,
where report.py reads them. Re-running --set overwrites that answer's grade.
"""
import argparse
import json
import sys
import textwrap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--set", action="append", default=[], metavar="Q/N=SCORE[f]")
    p.add_argument("--note", action="append", default=[], metavar="Q/N=TEXT")
    p.add_argument("--by", help="who graded (recorded with each grade)")
    o = p.parse_args()
    doc = json.load(open(o.file, encoding="utf-8"))
    if doc.get("test") != "quiz":
        sys.exit("not a quiz result")
    data = doc["data"]
    grades = {(g["q"], g["sample"]): g for g in data.get("grades", [])}

    if not o.set and not o.note:
        for a in data["answers"]:
            g = grades.get((a["q"], a["sample"]))
            print(f"===== {a['q']}/{a['sample']}  {a['gen_tokens']} tokens, finish={a['finish']}"
                  + (f"  GRADE {g['score']}{'f' if g['fabricated'] else ''} {g.get('note', '')}" if g else ""))
            print("KEY:", textwrap.fill(data["key"][a["q"]], 110, subsequent_indent="     "))
            print()
            print(a["content"])
            print()
        return

    for item in o.set:
        qn, val = item.split("=", 1)
        q, n = qn.split("/")
        val = val.strip().lower()
        g = grades.setdefault((q, int(n)), {"q": q, "sample": int(n)})
        g["fabricated"] = val.endswith("f")
        g["score"] = int(val.rstrip("f"))
        if g["score"] not in (0, 1, 2):
            sys.exit(f"score must be 0, 1 or 2: {item}")
        if o.by:
            g["by"] = o.by
    for item in o.note:
        qn, text = item.split("=", 1)
        q, n = qn.split("/")
        grades.setdefault((q, int(n)), {"q": q, "sample": int(n)})["note"] = text

    keys = [(a["q"], a["sample"]) for a in data["answers"]]
    data["grades"] = [grades[k] for k in keys if k in grades]
    done = all(k in grades and "score" in grades[k] for k in keys)
    m = doc["metrics"]
    m["graded"] = done
    if done:
        m["score"] = sum(grades[k]["score"] for k in keys)
        m["score_max"] = 2 * len(keys)
        m["fabrications"] = sum(bool(grades[k]["fabricated"]) for k in keys)
        per_q = {}
        for q, n in keys:
            per_q.setdefault(q, []).append(grades[(q, n)]["score"])
        m["per_question"] = per_q
    with open(o.file, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("graded" if done else f"{sum(1 for k in keys if k in grades and 'score' in grades[k])}/{len(keys)} graded",
          f"score {m.get('score')}/{m.get('score_max')} fabrications {m.get('fabrications')}" if done else "")


if __name__ == "__main__":
    main()
