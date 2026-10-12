"""Transcribe a vocal stem into word-timed lyrics with Whisper large-v3.

Usage:
    python stem_test/transcribe.py lead_vocals.wav --out lead.json [--language en]

Writes {"language": ..., "lines": [{"start", "end", "text", "words": [{"w", "s", "e", "p"}]}]}.
"""

import argparse
import json
import subprocess
import time

import numpy as np
from faster_whisper import WhisperModel


def load_16k_mono(path):
    # Decode with ffmpeg rather than faster-whisper's PyAV path, which breaks across PyAV releases.
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default=None)
    ap.add_argument("--prompt", default=None, help="optional context, e.g. song title and artist")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4,
                         download_root=args.cache_dir)
    start = time.perf_counter()
    segments, info = model.transcribe(
        load_16k_mono(args.audio),
        language=args.language,
        initial_prompt=args.prompt,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        # Sung lyrics repeat a lot; carrying text forward makes Whisper loop or invent lines.
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
    )

    lines = []
    for seg in segments:
        words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 2)}
                 for w in (seg.words or []) if w.word.strip()]
        if not words:
            continue
        lines.append({"start": words[0]["s"], "end": words[-1]["e"], "text": seg.text.strip(),
                      "avg_logprob": round(seg.avg_logprob, 2), "words": words})
        print(f"[{words[0]['s']:7.2f} - {words[-1]['e']:7.2f}] {seg.text.strip()}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"language": info.language, "language_probability": round(info.language_probability, 2),
                   "seconds": round(time.perf_counter() - start, 1), "lines": lines}, f, indent=1)
    print(f"{len(lines)} lines, language {info.language} ({info.language_probability:.2f}), "
          f"{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
