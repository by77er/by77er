"""Tighten word timings by forced-aligning known lyrics to a vocal stem.

Whisper's word timestamps come from attention and are often off by a few hundred ms on sung
vocals. This re-times each line with a CTC forced aligner (torchaudio's MMS_FA), searching a
window around the line's existing timing, and keeps the old timing for any line the aligner
is not confident about.

Usage:
    python stem_test/forced_align.py lyrics.json lead.wav --part lead --out lyrics_fa.json
"""

import argparse
import json
import re
import subprocess

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA

SR = MMS_FA.sample_rate
PAD = 0.8            # seconds of search window either side of a line's current timing
MIN_SCORE = 0.15     # mean per-character probability below which a line keeps its old timing
MAX_SHIFT = 3.0      # a word moving further than this from the old timing is treated as a misfire...
MAX_SHIFT_AGREED = 6.0  # ...unless the onset check agrees the new timing is at least as good
VOICED_DB = -40      # stem level (dBFS) above which the singer counts as audible


def load(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


def romanize(word):
    return re.sub(r"[^a-z']", "", word.lower())


class Aligner:
    def __init__(self):
        self.model = MMS_FA.get_model(with_star=False).eval()
        self.dictionary = MMS_FA.get_dict(star=None)

    @torch.inference_mode()
    def align(self, wave, words):
        """Return [(start_s, end_s, score)] per word, relative to the start of wave, or None."""
        tokens = [[self.dictionary[c] for c in romanize(w) if c in self.dictionary] for w in words]
        if any(not t for t in tokens):
            return None
        emission, _ = self.model(wave.unsqueeze(0))
        flat = [t for ws in tokens for t in ws]
        if emission.shape[1] < len(flat):
            return None
        labels, scores = F.forced_align(emission, torch.tensor([flat], dtype=torch.int32), blank=0)
        spans = F.merge_tokens(labels[0], scores[0].exp())
        ratio = wave.shape[0] / emission.shape[1] / SR
        out, i = [], 0
        for t in tokens:
            ws = spans[i:i + len(t)]
            i += len(t)
            out.append((ws[0].start * ratio, ws[-1].end * ratio, float(np.mean([s.score for s in ws]))))
        return out


def level_db(wave, hop=0.01):
    n = int(SR * hop)
    x = wave.numpy()[: len(wave) // n * n].reshape(-1, n)
    return 10 * np.log10((x ** 2).mean(1) + 1e-12), hop


def onset_fit(starts, env):
    """Share of word starts where the singer is audible just after the start and the level is rising."""
    db, hop = env
    good = 0
    for s in starts:
        i = int(s / hop)
        after, before = db[i:i + 15], db[max(0, i - 15):i]
        if len(after) and np.median(after) > VOICED_DB and (not len(before) or after.mean() >= before.mean() - 3):
            good += 1
    return good / max(1, len(starts))


def held_until(t, limit, env, quiet=0.15):
    """First time after t where the stem stays below VOICED_DB for `quiet` seconds (capped at limit)."""
    db, hop = env
    i, stop, run = int(t / hop), int(limit / hop), 0
    while i < stop and i < len(db):
        run = run + 1 if db[i] < VOICED_DB else 0
        if run * hop >= quiet:
            return max(t, (i - run + 1) * hop)
        i += 1
    return limit


def snap_line_starts(lines, env, stats, reach=1.5):
    """Move a line's first word onto the vocal onset that ends the silence before it.

    Entries after a rest are where timing errors are most visible, and where the audio gives an
    unambiguous answer: the level jumps from silence to singing.
    """
    db, hop = env
    snapped = 0
    for n, line in enumerate(lines):
        w = line["words"][0]
        lo = max(int((w["s"] - reach) / hop), int((lines[n - 1]["words"][-1]["s"] + 0.1) / hop) if n else 0, 15)
        hi = min(int((w["s"] + reach) / hop), int(w["e"] / hop), len(db) - 10)
        onsets = [i for i in range(lo, hi)
                  if np.median(db[i:i + 10]) > VOICED_DB and np.median(db[i:i + 10]) - db[i - 15:i].max() >= 12
                  and db[i] > db[i - 1]]
        if not onsets:
            continue
        i = min(onsets, key=lambda k: abs(k * hop - w["s"]))
        new_s = round(i * hop, 2)
        if abs(new_s - w["s"]) > 0.05:
            w["s"] = new_s
            line["start"] = new_s
            snapped += 1
    stats["snapped"] = snapped


def refine(lines, wave, aligner):
    total = wave.shape[0] / SR
    env = level_db(wave)
    stats = {"aligned": 0, "kept": 0, "shifts": [], "fit_before": [], "fit_after": []}
    orig = [(l["start"], l["end"]) for l in lines]  # search windows come from the timing we started with
    for n, line in enumerate(lines):
        prev_end = orig[n - 1][1] if n else 0.0
        next_start = orig[n + 1][0] if n + 1 < len(lines) else total
        a = max(0.0, line["start"] - PAD, prev_end - 0.3)
        b = min(total, line["end"] + PAD, next_start + 0.3)
        words = line["words"]
        if b - a < 0.5:
            stats["kept"] += 1
            continue
        res = aligner.align(wave[int(a * SR):int(b * SR)], [w["w"] for w in words])
        old_fit = onset_fit([w["s"] for w in words], env)
        if res is None:
            stats["kept"] += 1
            stats["fit_before"].append(old_fit); stats["fit_after"].append(old_fit)
            continue
        new = []
        for (s, e, sc), w in zip(res, words):
            # A short word spread over a long span usually means the aligner latched onto the tail
            # of the previous note; the sung onset is just before the word's last letter.
            longest = 0.15 + 0.1 * len(romanize(w["w"]))
            new.append((a + max(s, e - longest), a + e, sc))
        score = float(np.mean([sc for _, _, sc in new]))
        shifts = [abs(ns - w["s"]) for (ns, _, _), w in zip(new, words)]
        new_fit = onset_fit([s for s, _, _ in new], env)
        stats["fit_before"].append(old_fit)
        # The onset check can prefer a start on the tail of the previous note, so only let it veto
        # the aligner when the new timing is clearly worse.
        too_far = max(shifts) > (MAX_SHIFT_AGREED if new_fit >= old_fit else MAX_SHIFT)
        if score < MIN_SCORE or too_far or new_fit < old_fit - 0.34:
            stats["kept"] += 1
            stats["fit_after"].append(old_fit)
            line["fa"] = {"kept": True, "score": round(score, 2), "max_shift": round(max(shifts), 2),
                          "fit_old": round(old_fit, 2), "fit_new": round(new_fit, 2)}
            continue
        stats["fit_after"].append(new_fit)
        for k, (w, (s, e, sc)) in enumerate(zip(words, new)):
            stats["shifts"].append(s - w["s"])
            nxt = new[k + 1][0] if k + 1 < len(new) else None
            # CTC marks a held vowel as blank, so a word's span ends early. Run the fill on while
            # the singer is still audible, up to the next word.
            limit = nxt if nxt is not None else min(total, e + 4.0)
            end = held_until(e, limit, env)
            w["s"], w["e"], w["fa"] = round(s, 2), round(max(end, s + 0.08), 2), round(sc, 2)
        line["start"], line["end"] = words[0]["s"], words[-1]["e"]
        line["fa"] = {"kept": False, "score": round(score, 2), "fit_old": round(old_fit, 2), "fit_new": round(new_fit, 2)}
        stats["aligned"] += 1
    snap_line_starts(lines, env, stats)
    # a line's last word may be held into the gap, but not past the start of the next line
    for line, nxt in zip(lines, lines[1:]):
        last = line["words"][-1]
        last["e"] = round(max(last["s"] + 0.08, min(last["e"], nxt["start"] - 0.05)), 2)
        line["end"] = last["e"]
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyrics")
    ap.add_argument("audio", help="isolated vocal stem for this part")
    ap.add_argument("--part", default="lead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lyrics = json.load(open(args.lyrics))
    wave = load(args.audio)
    stats = refine(lyrics[args.part], wave, Aligner())
    json.dump(lyrics, open(args.out, "w"), indent=1)
    sh = np.array(stats["shifts"])
    print(f"{args.part}: {stats['aligned']} lines re-timed, {stats['kept']} kept, {stats['snapped']} line starts snapped to onsets; word starts on a sung onset: "
          f"{np.mean(stats['fit_before']):.0%} before -> {np.mean(stats['fit_after']):.0%} after")
    if len(sh):
        print(f"word start shift: median {np.median(sh) * 1000:+.0f} ms, "
              f"median |shift| {np.median(np.abs(sh)) * 1000:.0f} ms, 90th pct |shift| {np.percentile(np.abs(sh), 90) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
