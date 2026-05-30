# EE/CS 148B HW 3 — Vision-Language Models

Starter code for HW 3 of EE/CS 148B (Spring 2026). See `hw3.pdf` (in the assignment release) for the writeup.

## Repository Layout

```
hw3/
├── basics/                  # Shared building blocks (mostly provided)
│   ├── __init__.py
│   ├── model.py             # PROVIDED: Head, MultiHeadAttention, MLP, Block
│   ├── text_encoder.py      # PROVIDED: FrozenTextEncoder wrapper
│   ├── vit.py               # TODO: PatchEmbeddings, ViT  (§2)
│   ├── lora.py              # TODO: LoRALinear, apply_lora_to_attention  (§4)
│   └── rope.py              # TODO: RoPE1D, RoPE2D  (§6)
│
├── vlm/                     # VLM-specific code
│   ├── __init__.py
│   ├── clip.py              # TODO: clip_loss, projection heads  (§3)
│   ├── projector.py         # TODO: VisionLanguageProjector  (§5)
│   ├── model.py             # TODO: VisionLanguageModel (fusion + injection)  (§5)
│   ├── masking.py           # PROVIDED: 4D attention mask helpers
│   ├── data.py              # PROVIDED: EuroSAT / RESISC45 / CLEVR loaders
│   └── eval.py              # PROVIDED: zero-shot accuracy, CLEVR exact match
│
├── tests/                   # Test infrastructure
│   ├── adapters.py          # TODO: bind your implementations to run_* hooks
│   ├── test_vit.py
│   ├── test_clip.py
│   ├── test_lora.py
│   └── test_rope.py
│
├── scripts/                 # CLI entry points
│   ├── pretrain_clip.py     # §3.3 — CLIP-style pretraining on EuroSAT
│   ├── finetune_resisc.py   # §4.2 — full FT vs LoRA vs linear probe
│   ├── train_vlm.py         # §5    — VLM training on CLEVR
│   └── eval_vlm.py          # §5    — qualitative + exact-match eval
│
├── configs/                 # Hyperparameter configs (YAML)
│   ├── clip_eurosat.yaml
│   ├── lora_resisc.yaml
│   └── vlm_clevr.yaml
│
├── data/                    # Dataset cache (gitignored)
│   └── README.md
│
├── pyproject.toml
├── README.md
└── .gitignore
```

`PROVIDED` files contain working code and are not meant to be modified.
`TODO` files contain skeleton classes/functions with `raise NotImplementedError` and clear docstrings — these are what you implement.

## Setup

We recommend using [`uv`](https://docs.astral.sh/uv/) for dependency management (the same tool you used for HW1 and HW2).

```bash
# Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies, including test dependencies
uv sync --extra test

# Optionally install FlashAttention-2 (required for the VLM in §5)
uv sync --extra flash

# Verify the import works
uv run python -c "import basics; import vlm; print('OK')"
```

If you are running on Colab, mount the repo and `uv sync` from the repo root before each session.

## Running Tests

After implementing the relevant pieces in `tests/adapters.py`, run the corresponding tests as described in the writeup:

```bash
# If you have not already installed test dependencies
uv sync --extra test

# §2 — ViT
uv run pytest -k test_patch_embeddings
uv run pytest -k test_vit

# §3 — CLIP
uv run pytest -k test_clip_loss

# §4 — LoRA
uv run pytest -k test_lora_linear
uv run pytest -k test_apply_lora

# §6 — RoPE
uv run pytest -k test_rope_1d
uv run pytest -k test_rope_2d
```

Run all tests at once:

```bash
uv run pytest
```

## Running the Experiments

Each section of the writeup maps to a script in `scripts/`. The scripts read hyperparameters from `configs/`, which you can override at the command line.

```bash
# §3.3 — CLIP pretraining on EuroSAT
uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml

# §4.2 — Full FT vs LoRA vs linear probe on RESISC45
uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml --method lora --rank 8

# §5 — VLM training on CLEVR (use the ViT from your EuroSAT CLIP run in §3 — same 64/8/384/6/6 arch)
uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
    --pretrained-vit runs/clip_eurosat/best.pt --injection all_patches --mask-mode image_bidir

# §5 — Three-way injection sweep (2000 steps each, projector-only) + printed table / discussion
uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
    --pretrained-vit runs/clip_eurosat/best.pt --all-injections

# §5 — Qualitative evaluation
uv run python scripts/eval_vlm.py --checkpoint runs/vlm_clevr/best.pt --num-examples 10
```

## Datasets

The starter code uses:
- **EuroSAT** via `datasets.load_dataset("blanchon/EuroSAT_RGB")` (~90 MB)
- **RESISC45** via `datasets.load_dataset("timm/resisc45")` (preprocessed subset, ~150 MB)
- **CLEVR** — preprocessed 10k subset with original CLEVR image resolution. Run `uv run python scripts/download_clevr.py` before VLM training. The script downloads the zip from Google Drive: https://drive.google.com/file/d/1KsswLqfYLl1d91pg5kGUgwtPslo8njTB/view?usp=sharing

## Submission

Submit two files to Gradescope:
- `writeup.pdf` — your typeset answers to the written questions.
- `code.zip` — push this repo to a **private** GitHub repository and submit it via Gradescope's GitHub integration.

## Compute

- Free-tier Colab (T4, L4): §2, §4 work fine.
- L4 / A100: §3 (CLIP pretraining).
- A100 / H100: §5 (VLM training), §6 (RoPE ablations). Each run is ~1 hour.

Keep your Colab Pro+ receipts for end-of-quarter reimbursement.

## Acknowledgments

Course staff: Aadarsh Sahoo, Ziqi Ma.
