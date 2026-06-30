"""Final archive build with optional frame selector sweep.

Reads the Stage 9 best checkpoint, runs the frame selector sweep (one
distortion eval per mode per pair), encodes everything into the HNS9
archive format, and writes <output_dir>/0.bin.
"""
import json
from pathlib import Path

import torch

from codec import build_archive, parse_archive
from data import EVAL_SIZE
from frame_selector import sweep_frame_selector, PALETTE_MODE_IDS
from score import total_video_bytes


def run_codec_stage(prev_stage_output_dir: Path, final_output_dir: Path,
                    video_path, device=None, skip_selector=False) -> dict:
    final_output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    decoder_sd = torch.load(prev_stage_output_dir / "decoder_f32.pt", map_location='cpu')
    latents = torch.load(prev_stage_output_dir / "latents_f32.pt", map_location='cpu')
    n_pairs = latents.shape[0]

    selector_indices = None
    n_modes = 0
    if not skip_selector:
        from data import precompute_targets
        from model import HNeRVDecoder
        distortion_net, seg_targets_hard, pose_targets, _, _ = precompute_targets(video_path, device)
        decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE).to(device)
        decoder.load_state_dict(decoder_sd)
        decoder.eval()
        selector_indices = sweep_frame_selector(
            decoder, latents, distortion_net,
            seg_targets_hard, pose_targets,
            device=device,
        )
        n_modes = len(PALETTE_MODE_IDS)
        del decoder, distortion_net, seg_targets_hard, pose_targets
        torch.cuda.empty_cache()

    print("[codec_stage] building archive (greedy tensor order search)...", flush=True)
    archive = build_archive(
        decoder_sd, latents,
        meta_dict={"n_pairs": n_pairs, "latent_dim": 28, "base_channels": 36,
                   "eval_size": list(EVAL_SIZE)},
        selector_indices=selector_indices,
        selector_n_modes=n_modes,
        search_order=True,
    )
    archive_bytes = len(archive)

    out_path = final_output_dir / "0.bin"
    with open(out_path, "wb") as f:
        f.write(archive)
    print(f"[codec_stage] archive: {archive_bytes:,} bytes → {out_path}", flush=True)

    # Verify round-trip.
    decoder_sd_dec, _, _, _ = parse_archive(archive)
    for name in decoder_sd:
        orig = decoder_sd[name].detach().cpu().float()
        ma = orig.abs().max().item()
        scale = ma / 127 if ma > 0 else 1.0
        orig_q = (orig / scale).round().clamp(-127, 127)
        dec = decoder_sd_dec[name].detach().cpu().float()
        dec_q = (dec / scale).round().clamp(-127, 127)
        if not torch.allclose(orig_q, dec_q):
            raise RuntimeError(f"Codec round-trip FAILED for {name}")
    print("[codec_stage] round-trip verified OK", flush=True)

    meta_path = prev_stage_output_dir / "best_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        meta['final_archive_bytes'] = archive_bytes
        meta['has_selector'] = selector_indices is not None
        with open(final_output_dir / "final_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return {
        'final_archive_bytes': archive_bytes,
        'archive_path': str(out_path),
    }
