"""9-stage HNeRV training pipeline targeting score < 0.192.

Stages 1-8 are the same 8-stage curriculum as hnerv_muon (PR #95).
Stage 8 is extended to 10,000 epochs (vs 5,000 in hnerv_muon) with a
lower LR floor (5e-7 vs 5e-6) to extract more from the Muon trajectory.
Stage 9 (NEW): freeze the decoder, optimise only the latents for 2,000
epochs. This removes cross-pair interference and lets each pair's latent
converge to its per-pair optimum against the fixed quantised decoder.
After Stage 9, the codec stage runs a frame-selector sweep and builds the
final HNS9 archive.

Estimated training time:
  A100 GPU (Colab Pro): ~23 hours (fits in one session)
  RTX 4060 Laptop:      ~7.5 days

Usage:
  python -m submissions.hnerv_stage9.src.train
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import torch

from data import get_default_video_path
from stages.common import train_stage
from stages import (
    stage1_v328_ce,
    stage2_v331_softplus,
    stage3_v332_smooth,
    stage4_v332_qat,
    stage5_c1a_l7,
    stage6_lambda_sweep,
    stage7_sigma_sweep,
    stage8_muon_finetune,
    codec_stage,
)
from stages.stage9_latent_polish import run_latent_polish


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = HERE.parent.parent / "ckpts" / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"Device: {device}", flush=True)

    video_path = get_default_video_path()
    shared_state = {}
    t0 = time.time()

    # Stages 1-8: same curriculum as hnerv_muon.
    prev = None
    builders = [
        stage1_v328_ce.make_config,
        stage2_v331_softplus.make_config,
        stage3_v332_smooth.make_config,
        stage4_v332_qat.make_config,
        stage5_c1a_l7.make_config,
        stage6_lambda_sweep.make_config,
        stage7_sigma_sweep.make_config,
        stage8_muon_finetune.make_config,   # extended to 10,000 ep (vs 5,000 in hnerv_muon)
    ]
    for i, build in enumerate(builders, start=1):
        stage_out = out_root / f"stage{i}"
        cfg = build(stage_out) if i == 1 else build(prev, stage_out)
        result = train_stage(cfg, device, video_path=video_path,
                             shared_state=shared_state)
        print(f"[Stage {i}] best={result['best_score']:.4f} at ep{result['best_ep']} "
              f"(archive {result['archive_size']:,} bytes)", flush=True)
        prev = stage_out

    # Stage 9: latent-only polishing (decoder frozen).
    stage9_out = out_root / "stage9"
    run_latent_polish(
        resume_from=prev,
        output_dir=stage9_out,
        device=device,
        video_path=video_path,
        epochs=2000,
        lr=1e-4,
        lr_floor=1e-7,
        batch_size=16,
        eval_every=25,
        ema_decay=0.999,
        shared_state=shared_state,
    )
    print(f"[Stage 9] latent polish done", flush=True)

    # Codec + frame selector.
    codec_out = out_root / "submission_archive"
    print(f"\n[codec] re-encoding from {stage9_out}", flush=True)
    r = codec_stage.run_codec_stage(
        stage9_out, codec_out, video_path, device=device, skip_selector=False)
    print(f"[codec] archive bytes: {r['final_archive_bytes']:,}", flush=True)
    print(f"\nTotal wallclock: {(time.time() - t0) / 3600:.1f} hr", flush=True)
    print(f"Final archive: {r['archive_path']}", flush=True)


if __name__ == "__main__":
    main()
