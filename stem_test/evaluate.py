"""Compare off-the-shelf vocal / instrumental separators on MUSDB18 clips.

Runs each model through python-audio-separator, then scores the vocal and
instrumental outputs against MUSDB ground truth with the MDX-challenge SDR
(10*log10(|s|^2 / |s - s_hat|^2), higher is better).

Usage:
    python stem_test/evaluate.py --musdb-root DIR --tracks 5 --out OUT_DIR
"""

import argparse
import json
import time
from pathlib import Path

import musdb
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

MODELS = {
    "mel_roformer": "vocals_mel_band_roformer.ckpt",
    "bs_roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "mdx_inst_hq3": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "htdemucs_ft": "htdemucs_ft.yaml",
}


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    num = np.sum(ref**2) + 1e-8
    den = np.sum((ref - est) ** 2) + 1e-8
    return 10 * np.log10(num / den)


def load(path):
    audio, _ = sf.read(path, always_2d=True)
    return audio


def stems_from_outputs(files):
    """Map separator outputs to (vocals, instrumental) arrays.

    Two-stem models emit Vocals + Instrumental/Other; Demucs emits four stems,
    so the non-vocal ones are summed into the instrumental.
    """
    vocals, inst = None, []
    for f in files:
        name = Path(f).stem.lower()
        if "(vocals)" in name:
            vocals = load(f)
        else:
            inst.append(load(f))
    return vocals, np.sum(inst, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb-root", required=True)
    ap.add_argument("--tracks", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default="stem_test/output")
    args = ap.parse_args()

    out = Path(args.out)
    mix_dir = out / "mixtures"
    mix_dir.mkdir(parents=True, exist_ok=True)

    db = musdb.DB(root=args.musdb_root, subsets="test", download=not Path(args.musdb_root, "test").exists())
    tracks = db.tracks[: args.tracks]
    refs = {}
    for t in tracks:
        sf.write(mix_dir / f"{t.name}.wav", t.audio, t.rate)
        refs[t.name] = (t.targets["vocals"].audio, t.targets["accompaniment"].audio)

    results = {}
    for key in args.models:
        model_dir = out / key
        model_dir.mkdir(parents=True, exist_ok=True)
        sep = Separator(output_dir=str(model_dir), model_file_dir=str(out / "models"))
        sep.load_model(MODELS[key])
        rows = []
        for t in tracks:
            start = time.perf_counter()
            files = sep.separate(str(mix_dir / f"{t.name}.wav"))
            elapsed = time.perf_counter() - start
            files = [model_dir / Path(f).name for f in files]
            vocals, inst = stems_from_outputs(files)
            ref_v, ref_i = refs[t.name]
            rows.append({
                "track": t.name,
                "vocals_sdr": sdr(ref_v, vocals),
                "inst_sdr": sdr(ref_i, inst),
                "seconds": elapsed,
                "audio_seconds": len(t.audio) / t.rate,
            })
            print(f"{key:14s} {t.name[:40]:40s} voc {rows[-1]['vocals_sdr']:6.2f}  "
                  f"inst {rows[-1]['inst_sdr']:6.2f}  {elapsed:5.1f}s", flush=True)
        results[key] = rows

    print("\nmodel           vocals SDR  inst SDR  realtime x")
    summary = {}
    for key, rows in results.items():
        v = float(np.median([r["vocals_sdr"] for r in rows]))
        i = float(np.median([r["inst_sdr"] for r in rows]))
        rtf = sum(r["seconds"] for r in rows) / sum(r["audio_seconds"] for r in rows)
        summary[key] = {"vocals_sdr_median": v, "inst_sdr_median": i, "cpu_seconds_per_audio_second": rtf}
        print(f"{key:14s} {v:10.2f} {i:9.2f} {rtf:10.2f}")

    (out / "results.json").write_text(json.dumps({"summary": summary, "per_track": results}, indent=2))


if __name__ == "__main__":
    main()
