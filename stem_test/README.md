# Vocal / instrumental separation tests

Off-the-shelf models run through
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
scored on CPU (4 cores, no GPU).

## Results

The first 10 tracks of the MUSDB18 test set (7-second preview clips). SDR is in dB,
and higher is better. It's the median across tracks of the MDX-challenge SDR. "CPU
s/audio s" is processing time per second of audio.

| Model | Vocals SDR | Instrumental SDR | CPU s / audio s |
|---|---|---|---|
| Mel-Band RoFormer (Kim) `vocals_mel_band_roformer.ckpt` | 11.78 | 15.71 | 1.2 |
| BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | **12.18** | **16.37** | 8.8 |
| Demucs v4 fine-tuned `htdemucs_ft.yaml` (4 stems, non-vocal stems summed) | 9.51 | 12.66 | 4.4 |
| MDX-Net `UVR-MDX-NET-Inst_HQ_3.onnx` | 7.26 | 12.01 | 1.0 |

The per-track numbers are in `results_musdb7s_10tracks.json`.

Mel-Band RoFormer is the default in `separate.py`. It's within about 0.5 dB of
BS-RoFormer and about 7x faster on CPU. Choose BS-RoFormer when quality matters
more than time.

Caveats:
- The speed column comes from 7-second clips. A full 4:45 song through BS-RoFormer
  took 1 h 51 min on the same CPU, about 23 s per audio second, so expect full songs
  to run slower than the table suggests.
- 10 clips of 7 seconds each is a small sample.
- Some of these models may have seen MUSDB18 during training, so absolute scores
  may be optimistic. The ranking is the useful part.

## Usage

```sh
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r stem_test/requirements.txt

# split your own songs into vocal and non-vocal files (named `(Vocals)` and `(Instrumental)` or `(Other)`, depending on the model)
python stem_test/separate.py song.mp3 --out stems/
python stem_test/separate.py song.mp3 --model model_bs_roformer_ep_317_sdr_12.9755.ckpt

# re-run the benchmark (musdb downloads the 7s sample set on first use)
python stem_test/evaluate.py --musdb-root data/musdb --tracks 10
```

Install torch, torchvision and torchaudio together from the same index. Mixing them
gives `operator torchvision::nms does not exist`.

Model weights download on first use and can be close to 1 GB. If a download is
cut off, audio-separator keeps the truncated file and fails later. Delete the
file and run again.
