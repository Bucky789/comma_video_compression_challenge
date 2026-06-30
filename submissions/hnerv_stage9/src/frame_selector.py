"""Per-pair frame transform palette and sweep.

Adapted from hnerv_fec6_fixed_huffman_k16/src/frame_selector.py (MIT).
31-mode palette of deterministic frame-0 (or frame-1) transforms.
The sweep evaluates all modes for each pair and picks the best one.
"""
from __future__ import annotations

import math
import struct

import torch
import torch.nn.functional as F


PALETTE_MODE_IDS = (
    "none",
    "frame0_luma_bias_-4",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_+2",
    "frame0_luma_bias_+4",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_2",
    "frame0_blue_chroma_amp_3",
    "frame0_roll_dx+1_dy+0",
    "frame0_roll_dx-1_dy+0",
    "frame0_roll_dx+0_dy+1",
    "frame0_roll_dx+0_dy-1",
    "frame1_rgb_bias_p2_m1_m1",
    "frame1_rgb_bias_m2_p1_p1",
    "frame1_luma_bias_-1",
    "frame1_blue_chroma_amp_3",
    "frame1_rgb_bias_p0_m1_p1",
    "frame1_rgb_bias_p0_p1_m1",
    "frame1_luma_bias_+1",
    "frame1_blue_chroma_amp_1",
    "frame1_luma_bias_-2",
)

MODE_PARAMS: dict[str, tuple[str, tuple[int, ...]]] = {
    "none": ("identity", ()),
    "frame0_luma_bias_-4": ("rgb_bias", (-4, -4, -4)),
    "frame0_luma_bias_-2": ("rgb_bias", (-2, -2, -2)),
    "frame0_luma_bias_-1": ("rgb_bias", (-1, -1, -1)),
    "frame0_luma_bias_+1": ("rgb_bias", (1, 1, 1)),
    "frame0_luma_bias_+2": ("rgb_bias", (2, 2, 2)),
    "frame0_luma_bias_+4": ("rgb_bias", (4, 4, 4)),
    "frame0_rgb_bias_p0_m1_p1": ("rgb_bias", (0, -1, 1)),
    "frame0_rgb_bias_p0_p1_m1": ("rgb_bias", (0, 1, -1)),
    "frame0_rgb_bias_p2_m1_m1": ("rgb_bias", (2, -1, -1)),
    "frame0_rgb_bias_m2_p1_p1": ("rgb_bias", (-2, 1, 1)),
    "frame0_rgb_bias_p0_m2_p2": ("rgb_bias", (0, -2, 2)),
    "frame0_rgb_bias_p0_p2_m2": ("rgb_bias", (0, 2, -2)),
    "frame0_rgb_bias_p4_m2_m2": ("rgb_bias", (4, -2, -2)),
    "frame0_rgb_bias_m4_p2_p2": ("rgb_bias", (-4, 2, 2)),
    "frame0_blue_chroma_amp_1": ("blue_chroma", (1,)),
    "frame0_blue_chroma_amp_2": ("blue_chroma", (2,)),
    "frame0_blue_chroma_amp_3": ("blue_chroma", (3,)),
    "frame0_roll_dx+1_dy+0": ("roll", (1, 0)),
    "frame0_roll_dx-1_dy+0": ("roll", (-1, 0)),
    "frame0_roll_dx+0_dy+1": ("roll", (0, 1)),
    "frame0_roll_dx+0_dy-1": ("roll", (0, -1)),
    "frame1_rgb_bias_p2_m1_m1": ("rgb_bias", (2, -1, -1)),
    "frame1_rgb_bias_m2_p1_p1": ("rgb_bias", (-2, 1, 1)),
    "frame1_luma_bias_-1": ("rgb_bias", (-1, -1, -1)),
    "frame1_blue_chroma_amp_3": ("blue_chroma", (3,)),
    "frame1_rgb_bias_p0_m1_p1": ("rgb_bias", (0, -1, 1)),
    "frame1_rgb_bias_p0_p1_m1": ("rgb_bias", (0, 1, -1)),
    "frame1_luma_bias_+1": ("rgb_bias", (1, 1, 1)),
    "frame1_blue_chroma_amp_1": ("blue_chroma", (1,)),
    "frame1_luma_bias_-2": ("rgb_bias", (-2, -2, -2)),
}

CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512


def _blue_tile(height: int, width: int, *, device, dtype) -> torch.Tensor:
    tile = torch.tensor([
        [-1, 1, -1, 1, 1, -1, 1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [-1, 1, 1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [-1, 1, 1, -1, -1, 1, 1, -1],
    ], dtype=dtype, device=device)
    rh = (height + 7) // 8
    rw = (width + 7) // 8
    return tile.repeat(rh, rw)[:height, :width]


def apply_mode(frame_chw: torch.Tensor, mode_id: str) -> torch.Tensor:
    family, params = MODE_PARAMS[mode_id]
    if family == "identity":
        return frame_chw
    out = frame_chw.clone()
    if family == "rgb_bias":
        delta = torch.tensor(params, dtype=out.dtype, device=out.device).view(3, 1, 1)
        return out + delta
    if family == "blue_chroma":
        amp = float(params[0])
        _, H, W = out.shape
        tile = _blue_tile(H, W, device=out.device, dtype=out.dtype)
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
        return out
    if family == "roll":
        dx, dy = int(params[0]), int(params[1])
        return torch.roll(out, shifts=(dy, dx), dims=(1, 2))
    raise ValueError(f"unsupported mode family {family!r}")


def apply_selector_to_frames(
    frames_bchw: torch.Tensor,
    selector_indices: list[int],
    *,
    pair_start: int = 0,
) -> torch.Tensor:
    """Apply per-pair selector modes to a flat (n_frames, 3, H, W) batch."""
    out = frames_bchw.clone()
    n_pairs = frames_bchw.shape[0] // 2
    for offset in range(n_pairs):
        pair_index = pair_start + offset
        if pair_index >= len(selector_indices):
            break
        mode_id = PALETTE_MODE_IDS[int(selector_indices[pair_index])]
        if mode_id == "none":
            continue
        frame_offset = offset * 2 + (1 if mode_id.startswith("frame1_") else 0)
        out[frame_offset] = apply_mode(out[frame_offset], mode_id)
    return out.clamp_(0.0, 255.0).round_()


@torch.inference_mode()
def sweep_frame_selector(
    decoder,
    latents: torch.Tensor,
    distortion_net,
    seg_targets_hard: torch.Tensor,
    pose_targets: torch.Tensor,
    *,
    device: torch.device,
) -> list[int]:
    """For each pair, sweep all 31 modes and pick the one with lowest distortion.

    Returns a list of mode indices (one per pair).
    """
    n_pairs = latents.shape[0]
    n_modes = len(PALETTE_MODE_IDS)
    decoder.eval()

    best_indices = []
    print(f"[frame_selector] sweeping {n_modes} modes × {n_pairs} pairs...", flush=True)

    for pair_idx in range(n_pairs):
        z = latents[pair_idx:pair_idx + 1].to(device)
        decoded = decoder(z)  # (1, 2, 3, EVAL_H, EVAL_W)
        flat = decoded.reshape(2, 3, EVAL_H, EVAL_W)
        with torch.autocast(device.type, dtype=torch.float16):
            up_f16 = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode='bicubic', align_corners=False)
        up = up_f16.float()
        up_rounded = up.clamp(0, 255).round()  # (2, 3, H, W)

        best_mode = 0
        best_score = float('inf')

        for mode_idx, mode_id in enumerate(PALETTE_MODE_IDS):
            if mode_id == "none":
                candidate = up_rounded
            else:
                frame_slot = 1 if mode_id.startswith("frame1_") else 0
                candidate = up_rounded.clone()
                candidate[frame_slot] = apply_mode(up_rounded[frame_slot], mode_id).clamp(0, 255).round()

            # Evaluate distortion (candidate is (2, 3, H, W) uint8-range float).
            candidate_bhwc = candidate.permute(0, 2, 3, 1).unsqueeze(0)  # (1, 2, H, W, 3)
            candidate_bhwc = candidate_bhwc.clamp(0, 255)

            with torch.autocast(device.type, dtype=torch.float16):
                posenet_in, segnet_in = distortion_net.preprocess_input(candidate_bhwc)
                seg_logits = distortion_net.segnet(segnet_in)
                pose_out = distortion_net.posenet(posenet_in)
            seg_logits = seg_logits.float()
            pose_out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in pose_out.items()}

            # seg disagreement + sqrt(10 * pose_mse)
            seg_pred = seg_logits.argmax(dim=1)  # (1, H, W)
            seg_target = seg_targets_hard[pair_idx:pair_idx + 1].long().to(device)
            seg_d = (seg_pred != seg_target).float().mean().item()
            pose_mse = F.mse_loss(pose_out['pose'][:, :6],
                                  pose_targets[pair_idx:pair_idx + 1]).item()
            combined = 100.0 * seg_d + (10.0 * pose_mse) ** 0.5

            if combined < best_score:
                best_score = combined
                best_mode = mode_idx

        best_indices.append(best_mode)

        if (pair_idx + 1) % 100 == 0:
            chosen = PALETTE_MODE_IDS[best_mode]
            print(f"  pair {pair_idx+1}/{n_pairs}: best={chosen!r} score={best_score:.4f}",
                  flush=True)

    n_non_none = sum(1 for i in best_indices if i != 0)
    print(f"[frame_selector] {n_non_none}/{n_pairs} pairs got non-identity transform",
          flush=True)
    return best_indices
