#!/usr/bin/env python3
"""HNeRV Stage 9 inflation: decode archive → raw RGB frames.

Usage:
  python inflate.py <src.bin> <dst.raw>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_archive
from model import HNeRVDecoder
from frame_selector import apply_selector_to_frames

CAMERA_H, CAMERA_W = 874, 1164


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta, selector_indices = parse_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = int(meta["n_pairs"])
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                               mode="bicubic", align_corners=False)
            rounded = up.clamp(0, 255).round()
            if selector_indices is not None:
                rounded = apply_selector_to_frames(rounded, selector_indices, pair_start=i)
            frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
