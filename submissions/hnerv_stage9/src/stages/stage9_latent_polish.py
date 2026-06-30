"""Stage 9: latent_polish — decoder frozen, latents only.

Key insight: after joint training, each pair's latent is at the global
optimum of a joint objective that compromises across 600 pairs. With the
decoder fixed, each latent can be independently tuned to its own per-pair
minimum. This removes cross-pair interference and lets hard pairs recover.

Optimizer: AdamW on latents only. Decoder is frozen (no gradients).
QAT is still applied to decoder weights during forward (so latents are
optimised against the quantised decoder, matching what inflate.py does).
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))

from model import HNeRVDecoder
from losses import l7_softplus_seg_loss, apply_qat, restore_qat, ema_update
from data import precompute_targets, EVAL_SIZE
from codec import build_archive, parse_archive
from score import evaluate_decoder, compute_score, total_video_bytes


def run_latent_polish(
    resume_from: Path,
    output_dir: Path,
    device: torch.device,
    video_path=None,
    epochs: int = 2000,
    lr: float = 1e-4,
    lr_floor: float = 1e-7,
    batch_size: int = 8,
    eval_every: int = 25,
    ema_decay: float = 0.999,
    shared_state: dict | None = None,
):
    """Freeze the decoder from resume_from and optimise only the latents."""
    if video_path is None:
        from data import get_default_video_path
        video_path = get_default_video_path()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}\n[stage9_latent_polish] {epochs} ep, lr={lr}, "
          f"batch={batch_size}\n{'='*80}", flush=True)

    # Load decoder and freeze it.
    decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE).to(device)
    decoder.load_state_dict(torch.load(resume_from / "final_decoder.pt", map_location=device))
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # Load latents as a trainable parameter.
    latents = nn.Parameter(torch.load(resume_from / "final_latents.pt", map_location=device))
    ema_latents = latents.data.clone()

    # Precompute targets once.
    if shared_state and 'distortion_net' in shared_state and shared_state.get('video_path') == video_path:
        distortion_net = shared_state['distortion_net']
        seg_targets_hard = shared_state['seg_targets_hard']
        pose_targets = shared_state['pose_targets']
        n_pairs = shared_state['n_pairs']
    else:
        distortion_net, seg_targets_hard, pose_targets, _, n_pairs = (
            precompute_targets(video_path, device))
        if shared_state is not None:
            shared_state.update({
                'distortion_net': distortion_net,
                'seg_targets_hard': seg_targets_hard,
                'pose_targets': pose_targets,
                'n_pairs': n_pairs,
                'video_path': video_path,
            })

    opt = torch.optim.AdamW([latents], lr=lr, weight_decay=0.0)
    eta_min_ratio = max(lr_floor / lr, 1e-3)
    def lr_lambda(ep):
        return max(0.5 * (1 + math.cos(math.pi * ep / epochs)), eta_min_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tvb = total_video_bytes(video_path)
    seg_loss_fn = lambda logits, targets: l7_softplus_seg_loss(
        logits, targets, tau=0.3, l7_threshold=1.0, l7_mult=4.0)

    best_score = float('inf')
    best_ep = 0
    t0 = time.time()

    for epoch in range(epochs):
        pair_indices = torch.randperm(n_pairs, device=device)
        epoch_loss = 0.0
        nb = 0

        for batch_start in range(0, n_pairs, batch_size):
            idx = pair_indices[batch_start:batch_start + batch_size]
            B = len(idx)

            # Apply fake quantization to frozen decoder weights so latents
            # are tuned to the quantised model (matches inflate.py behaviour).
            # NOTE: we do NOT use torch.no_grad() here — gradients must flow
            # through the decoder to the latents. Decoder params have
            # requires_grad=False so only latents accumulate gradients.
            originals = apply_qat(decoder)
            decoded_pair = decoder(latents[idx])
            restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
            with torch.autocast('cuda', dtype=torch.float16):
                up = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
                down = F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)
            decoded_bhwc = down.float().reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
            decoded_clamped = decoded_bhwc.clamp(0, 255)
            decoded_bhwc = decoded_clamped + (decoded_clamped.round() - decoded_clamped).detach()

            with torch.autocast('cuda', dtype=torch.float16):
                posenet_in, segnet_in = distortion_net.preprocess_input(decoded_bhwc)
                seg_out = distortion_net.segnet(segnet_in)
                pose_out = distortion_net.posenet(posenet_in)
            seg_out = seg_out.float()

            seg_l = seg_loss_fn(seg_out, seg_targets_hard[idx.cpu()].long().to(device))
            pose_mse = F.mse_loss(pose_out['pose'][:, :6].float(), pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            loss = 100.0 * seg_l + 1.0 * pose_l

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([latents], 1.0)
            opt.step()

            with torch.no_grad():
                ema_latents.mul_(ema_decay).add_(latents.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            nb += 1

        sched.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [stage9] ep{epoch+1}/{epochs} loss={epoch_loss/nb:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if (epoch + 1) % eval_every == 0:
            # Build archive with frozen decoder + polished EMA latents.
            decoder_sd = {k: v for k, v in decoder.state_dict().items()}
            archive = build_archive(
                decoder_sd, ema_latents.cpu(),
                meta_dict={"n_pairs": n_pairs, "latent_dim": 28, "base_channels": 36,
                           "eval_size": list(EVAL_SIZE)})
            archive_size = len(archive)
            eval_decoder_sd, eval_lat, _, _ = parse_archive(archive)
            eval_dec = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE).to(device)
            eval_dec.load_state_dict(eval_decoder_sd)
            eval_dec.eval()
            dist = evaluate_decoder(eval_dec, eval_lat.to(device), distortion_net,
                                    video_path, batch_pairs=8, device=device)
            result = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                   archive_size, tvb)
            del eval_dec
            torch.cuda.empty_cache()

            print(f"    >>> ep{epoch+1}: score={result['score']:.4f} "
                  f"seg={result['seg_distortion']:.5f} pose={result['pose_distortion']:.6f} "
                  f"arch={archive_size:,}", flush=True)

            if result['score'] < best_score:
                best_score = result['score']
                best_ep = epoch + 1
                with open(output_dir / "best_archive.bin", "wb") as f:
                    f.write(archive)
                # Save the FROZEN decoder + best EMA latents for codec stage.
                torch.save(decoder.state_dict(), output_dir / "decoder_f32.pt")
                torch.save(ema_latents.cpu(), output_dir / "latents_f32.pt")
                with open(output_dir / "best_meta.json", "w") as f:
                    json.dump({"stage": "stage9_latent_polish",
                               "score": result['score'],
                               "seg_distortion": result['seg_distortion'],
                               "pose_distortion": result['pose_distortion'],
                               "archive_bytes": archive_size,
                               "epoch": epoch + 1}, f, indent=2)

    # Also write final latents for any downstream stages.
    torch.save(decoder.state_dict(), output_dir / "final_decoder.pt")
    torch.save(ema_latents.cpu(), output_dir / "final_latents.pt")

    print(f"\n[stage9_latent_polish] BEST: {best_score:.4f} at ep{best_ep}", flush=True)
    return {
        'stage': 'stage9_latent_polish',
        'best_score': best_score,
        'best_ep': best_ep,
        'output_ckpt_dir': output_dir,
    }
