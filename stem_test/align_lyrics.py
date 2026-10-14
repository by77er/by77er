"""Snap a Whisper transcript onto known lyrics, keeping Whisper's word timings.

Takes the word-timed JSON from transcribe.py and a text file with one lyric line per line
(as sung, repeats included). Words are matched in order with difflib; each matched word takes
its text from the lyrics file and its timing from the transcript, and lines follow the lyrics
file. Anything that can't be matched one-to-one is reported so it can be fixed by hand.

Usage:
    python stem_test/align_lyrics.py lead.json lyrics.txt --out lead_aligned.json
"""

import argparse
import difflib
import json
import re

MAX_WORD_SECS = 1.6  # Whisper stretches words over held notes and gaps; cap for display


def norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def align(lines, canon):
    words = [dict(w) for l in lines for w in l["words"]]
    for w in words:
        w["e"] = min(w["e"], w["s"] + MAX_WORD_SECS)
    ctoks = [(li, t) for li, line in enumerate(canon) for t in line.split()]
    sm = difflib.SequenceMatcher(None, [norm(w["w"]) for w in words], [norm(t) for _, t in ctoks], autojunk=False)
    groups, problems = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                li, text = ctoks[j1 + k]
                groups.setdefault(li, []).append({**words[i1 + k], "w": text})
        else:
            problems.append({"heard": [w["w"] for w in words[i1:i2]], "lyrics": [t for _, t in ctoks[j1:j2]],
                             "at": words[i1]["s"] if i1 < len(words) else None})
    out = []
    for li in sorted(groups):
        ws = groups[li]
        if [w["w"] for w in ws] != canon[li].split():
            problems.append({"partial_line": canon[li], "got": [w["w"] for w in ws], "at": ws[0]["s"]})
        out.append({"start": ws[0]["s"], "end": ws[-1]["e"], "text": " ".join(w["w"] for w in ws), "words": ws})
    missing = [canon[i] for i in range(len(canon)) if i not in groups]
    return out, problems, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("lyrics")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = json.load(open(args.transcript))["lines"]
    canon = [l.strip() for l in open(args.lyrics) if l.strip()]
    out, problems, missing = align(lines, canon)
    json.dump({"lines": out}, open(args.out, "w"), indent=1)
    print(f"{len(out)}/{len(canon)} lines aligned")
    for p in problems:
        print("check:", p)
    for m in missing:
        print("not sung / not heard:", m)


if __name__ == "__main__":
    main()
