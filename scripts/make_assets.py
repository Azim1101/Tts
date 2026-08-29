#!/usr/bin/env python3
"""
Convert the small Numpy feature files (mel_fb.npz, vocos_head.npz) shipped in the
HuggingFace repo into the raw little-endian binary format consumed by the app:

    mel_fb.bin
      int32[3]  n_fft, hop, n_mels
      float32[nFreq*nMels]  mel filterbank, (nMels x nFreq) row-major
      float32[n_fft]        analysis window (periodic Hann)

    vocos_head.bin
      int32[3]  n_fft, hop, win_length
      float32[1026*512]  linear_weight (out x in) row-major
      float32[1026]      linear_bias
      float32[1024]      synthesis window

The `.onnx` graphs are large and are NOT converted here — they are downloaded
verbatim by scripts/download_models.sh.

Usage:
    python3 scripts/make_assets.py --in <dir with npz> --out <dir for bin>
"""

import argparse
import os
import struct
import sys

import numpy as np


def _find(keys, *candidates):
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"none of {candidates} found among keys {keys}")


def convert_mel_fb(npz_path, out_path):
    d = np.load(npz_path)
    n_fft = int(d[_find(d.files, "n_fft", "N_FFT")])
    hop = int(d[_find(d.files, "hop", "hop_length", "HOP")])
    n_mels = int(d[_find(d.files, "n_mels", "N_MELS", "num_mels")])
    fb = np.asarray(d[_find(d.files, "fb", "mel_fb", "filterbank", "fb")], dtype=np.float32)
    win = np.asarray(d[_find(d.files, "window", "win", "hann", "window")], dtype=np.float32).reshape(-1)

    n_freq = n_fft // 2 + 1
    # Normalise so that fb is (nMels x nFreq) row-major.
    if fb.shape == (n_freq, n_mels):
        fb = fb.T
    if fb.shape != (n_mels, n_freq):
        raise ValueError(f"unexpected fb shape {fb.shape}, expected ({n_mels},{n_freq})")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", n_fft, hop, n_mels))
        f.write(fb.astype(np.float32).tobytes())
        f.write(win.astype(np.float32).tobytes())
    print(f"  wrote {out_path}  (n_fft={n_fft} hop={hop} n_mels={n_mels})")


def convert_vocos_head(npz_path, out_path):
    d = np.load(npz_path)
    n_fft = int(d[_find(d.files, "n_fft", "N_FFT")])
    hop = int(d[_find(d.files, "hop", "hop_length", "HOP")])
    win_length = int(d[_find(d.files, "win_length", "win_length", "win_len")])
    weight = np.asarray(d[_find(d.files, "linear_weight", "weight")], dtype=np.float32)
    bias = np.asarray(d[_find(d.files, "linear_bias", "bias")], dtype=np.float32).reshape(-1)
    win = np.asarray(d[_find(d.files, "window", "win", "hann", "window")], dtype=np.float32).reshape(-1)

    # weight expected (1026, 512) = (out, in).
    if weight.ndim != 2:
        raise ValueError(f"unexpected weight ndim {weight.ndim}")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", n_fft, hop, win_length))
        f.write(weight.astype(np.float32).tobytes())
        f.write(bias.astype(np.float32).tobytes())
        f.write(win.astype(np.float32).tobytes())
    print(f"  wrote {out_path}  (n_fft={n_fft} hop={hop} win_len={win_length})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    convert_mel_fb(os.path.join(args.in_dir, "mel_fb.npz"), os.path.join(args.out_dir, "mel_fb.bin"))
    convert_vocos_head(os.path.join(args.in_dir, "vocos_head.npz"), os.path.join(args.out_dir, "vocos_head.bin"))
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
