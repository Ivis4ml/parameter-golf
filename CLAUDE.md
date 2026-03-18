# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI's **Parameter Golf** challenge (2026-03-18 to 2026-04-30): train the lowest bits-per-byte (BPB) language model that fits in a **16MB artifact** (code bytes + compressed model bytes) and trains in under **10 minutes on 8×H100s**, evaluated on the FineWeb validation set.

Current SOTA: `1.2244` BPB (Naive Baseline). Lower is better.

## Running Locally (Apple Silicon / MLX)

```bash
# Download data (small subset for local smoke tests)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Quick smoke run
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

## Running on GPU (CUDA)

```bash
# Download full data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Single GPU
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8×H100 (leaderboard submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key env vars: `MAX_WALLCLOCK_SECONDS` (default 600), `VAL_LOSS_EVERY` (default 1000), `TRAIN_LOG_EVERY` (default 200). Set `MAX_WALLCLOCK_SECONDS=0` for unlimited compute runs.

## Architecture Overview

`train_gpt.py` is the single-file submission script (hard limit: 1500 lines). All competitive submissions go in `/records/`, not in the root scripts.

### Model (`GPT` class, line ~648)

Transformer with **U-Net skip connections**: the first `num_layers//2` blocks act as an encoder (storing skip tensors), the second half as a decoder (consuming them in reverse order). Each `Block` has:
- Pre-norm attention (`CausalSelfAttention`): RMSNorm → RoPE → GQA → output proj
- Pre-norm MLP: ReLU² activation (`relu(x)²`)
- Per-dimension learned `attn_scale`, `mlp_scale`, and `resid_mix` scalars
- `q_gain` per-head learnable query scaling

Baseline config: 9 layers × 512 dim, 8 heads / 4 KV heads (GQA), MLP 2×, vocab 1024, tied embeddings, logit softcap (tanh).

### Optimizer Split

Three separate optimizers, each with its own LR:
- **Muon** (`MATRIX_LR=0.04`): 2D matrix params in transformer blocks (Q/K/V/proj/fc/mlp projections). Uses Newton-Schulz orthogonalization.
- **Adam** tok embedding (`TIED_EMBED_LR=0.05` if tied, else `EMBED_LR=0.6`)
- **Adam** scalars (`SCALAR_LR=0.04`): 1D params, control tensors (`attn_scale`, `resid_mix`, etc.), skip weights

LR schedule: linear warmdown over the last `WARMDOWN_ITERS=1200` steps (or last portion of wallclock time if capped). Muon momentum warmup: 0.85→0.95 over 500 steps.

### Artifact Size Accounting

Artifact = `len(train_gpt.py UTF-8 bytes)` + `len(zlib.compress(torch.save(int8_quantized_state_dict)))`. Must be ≤ 16,000,000 bytes (decimal, not MiB).

Post-training quantization: per-row int8 for 2D tensors, per-tensor int8 for others. Small tensors (≤65536 elements) and control tensors are kept in fp16/fp32. See `quantize_state_dict_int8` / `dequantize_state_dict_int8`.

### BPB Metric

BPB is tokenizer-agnostic: `val_loss (nats/token) / log(2) × (tokens / bytes)`. The bytes-per-token ratio is computed from SentencePiece piece lengths via lookup tables (`build_sentencepiece_luts`). The leaderboard score uses the **post-quantization roundtrip** BPB (`final_int8_zlib_roundtrip_exact val_bpb`).

### Data Layout

- `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` — training shards (uint16 tokens, 100M tokens/shard)
- `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin` — fixed first-50k-doc validation split
- `data/tokenizers/fineweb_1024_bpe.model` — SentencePiece BPE, vocab 1024

Shard format: 256-int32 header (magic=20240520, version=1, num_tokens), then raw uint16 token IDs.

## Submission Structure

New submissions are PRs adding a folder to `records/track_10min_16mb/` (or `records/track_non_record_16mb/` for unlimited compute). Required files: `README.md`, `submission.json`, `train.log`, `train_gpt.py`. The script must run standalone within the records folder.

To beat the current SOTA, the submission must improve BPB by ≥0.005 at `p < 0.01` (multiple run logs required).
