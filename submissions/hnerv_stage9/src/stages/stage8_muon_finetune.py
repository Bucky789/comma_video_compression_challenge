"""Stage 8: muon_finetune — Muon on hidden Conv2d weights, AdamW on rest.

Extended to 8,000 epochs (vs 5,000 in hnerv_muon) with a lower LR floor
(5e-7 vs 5e-6) to extract more from the Muon trajectory.
"""
from pathlib import Path

from .common import StageConfig
from losses import l7_softplus_seg_loss


def make_config(resume_from: Path, output_dir: Path, epochs: int = 10000,
                muon_weight_decay: float = 5e-4) -> StageConfig:
    return StageConfig(
        name="stage8_muon_finetune",
        seg_loss_fn=lambda logits, targets: l7_softplus_seg_loss(
            logits, targets, tau=0.3, l7_threshold=1.0, l7_mult=4.0),
        epochs=epochs,
        eval_every=25,
        batch_size=4,
        ema_decay=0.999,
        use_muon=True,
        adamw_lr=1e-5,
        muon_lr=2e-4,
        muon_weight_decay=muon_weight_decay,
        latent_lr_mult=10.0,
        grad_clip=1.0,
        grad_clip_muon=1.0,
        lr_floor_ratio=5e-7,    # lower floor than hnerv_muon (was 5e-6)
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.02,
        cat_sigma=0.1,
        use_qat=True,
        resume_from=resume_from,
        output_dir=output_dir,
    )
