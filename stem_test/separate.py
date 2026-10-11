"""Split audio files into vocal and instrumental stems.

Usage:
    python stem_test/separate.py song.mp3 [more files...] --out stems/
"""

import argparse

from audio_separator.separator import Separator

DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="stems")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--format", default="WAV", help="WAV, FLAC or MP3")
    args = ap.parse_args()

    sep = Separator(output_dir=args.out, output_format=args.format)
    sep.load_model(args.model)
    for path in args.inputs:
        for f in sep.separate(path):
            print(f)


if __name__ == "__main__":
    main()
