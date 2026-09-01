#!/usr/bin/env python3
"""Convert mel_fb.npz / vocos_head.npz into flat little-endian binaries that the
C++ engine can mmap without a numpy/zip dependency.

    python3 scripts/make_assets.py --in <src_dir_with_npz> --out <dst_dir>

Layouts (all little-endian):

  mel_fb.bin     "DVMF" | i32 version=1 | i32 n_fft | i32 hop | i32 n_mels
                        | i32 n_freqs | i32 n_cols
                        | f32 fb[n_freqs * n_cols]  (row-major)
                        | f32 window[n_fft]

  vocos_head.bin "DVVH" | i32 version=1 | i32 out_dim | i32 in_dim
                        | i32 n_fft | i32 hop_length | i32 win_length
                        | f32 W[out_dim * in_dim]   (row-major)
                        | f32 b[out_dim]
                        | f32 window[win_length]
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def convert(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    mel_npz = src_dir / "mel_fb.npz"
    if mel_npz.exists():
        z = np.load(mel_npz)
        fb = np.ascontiguousarray(z["fb"], dtype="<f4")
        win = np.ascontiguousarray(z["window"], dtype="<f4")
        with open(dst_dir / "mel_fb.bin", "wb") as f:
            f.write(b"DVMF")
            f.write(struct.pack("<6i", 1, int(z["n_fft"]), int(z["hop"]),
                                int(z["n_mels"]), fb.shape[0], fb.shape[1]))
            f.write(fb.tobytes())
            f.write(win.tobytes())
        print(f"  wrote {dst_dir / 'mel_fb.bin'}  ({(dst_dir / 'mel_fb.bin').stat().st_size} bytes)")

    head_npz = src_dir / "vocos_head.npz"
    if head_npz.exists():
        h = np.load(head_npz)
        W = np.ascontiguousarray(h["linear_weight"], dtype="<f4")
        b = np.ascontiguousarray(h["linear_bias"], dtype="<f4")
        hw = np.ascontiguousarray(h["window"], dtype="<f4")
        with open(dst_dir / "vocos_head.bin", "wb") as f:
            f.write(b"DVVH")
            f.write(struct.pack("<6i", 1, W.shape[0], W.shape[1], int(h["n_fft"]),
                                int(h["hop_length"]), int(h["win_length"])))
            f.write(W.tobytes())
            f.write(b.tobytes())
            f.write(hw.tobytes())
        print(f"  wrote {dst_dir / 'vocos_head.bin'}  ({(dst_dir / 'vocos_head.bin').stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default=".")
    ap.add_argument("--out", dest="out_dir", default=".")
    args = ap.parse_args()
    convert(Path(args.in_dir), Path(args.out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
