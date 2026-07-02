# hnerv_stage9

9-stage curriculum extending [hnerv_muon](https://github.com/commaai/comma_video_compression_challenge/pull/99) with two novel additions: a **Stage 9 latent polish** pass (gradient descent on all 28 latent dims against the frozen quantized decoder) and **LZMA-compressed latents with greedy Brotli tensor ordering** in the codec.

## Architecture

- **Decoder**: 229K-param HNeRV, 6 upsample blocks, 36 base channels
- **Latents**: 28-d per frame pair, trained end-to-end then polished in Stage 9
- **Codec**: INT8 quantized weights (Brotli, greedy tensor ordering) + LZMA latents + fp16 scales + FES1 frame selector (31 modes, 598/600 pairs non-identity)

## Training curriculum

| Stage | Loss | Epochs | Notes |
|-------|------|--------|-------|
| 1 | CE seg + pose | 2500 | Random init |
| 2 | τ-Softplus seg + pose | 2000 | |
| 3 | Smooth disagreement seg | 500 | Fresh LR |
| 4 | + QAT | 500 | INT8 STE |
| 5 | + L7-weighted + C1a entropy (λ=0.01) | 4000 | |
| 6 | λ sweep → 0.02 | 750 | |
| 7 | σ sweep → 0.1 | 750 | |
| 8 | + Muon on Conv2d weights | 5000 | Newton-Schulz orthogonalization |
| 9 | Latent polish (frozen decoder) | 2000 | **Novel: all 28 dims via AdamW** |

Stage 9 freezes the decoder and optimizes latents directly against the quantized forward pass — zero archive overhead, pure distortion reduction.

## Archive identity

| Field | Value |
|---|---|
| Archive bytes | `179,431` |
| ZIP members | 1 (`0.bin`) |
| Inflate runtime deps | `torch`, `brotli`, `safetensors`, `einops`, `timm` |
| Inflate GPU required | no (faster with CUDA) |

## Inflate

```bash
unzip archive.zip -d /tmp/data
echo "0.mkv" > /tmp/list.txt
bash inflate.sh /tmp/data /tmp/out /tmp/list.txt
```

## Compress (reproduce)

Train from scratch on an A100 (~34h):

```bash
# See colab_train.ipynb for the full Colab-ready training pipeline
python submissions/hnerv_stage9/src/train.py
```
