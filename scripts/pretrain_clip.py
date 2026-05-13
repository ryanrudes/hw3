"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import wandb
import math

from itertools import chain
from tqdm import tqdm

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler

from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from basics.vit import ViT
from basics.text_encoder import FrozenTextEncoder
from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
from vlm.eval import zeroshot_classification_accuracy

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in the training loop.
    # Sketch:
    #   1. Build train/val/test loaders via vlm.data.build_eurosat_loaders.
    #   2. Build the ViT (basics.vit.ViT) and FrozenTextEncoder.
    #   3. Build ProjectionHeads + logit_scale.
    #   4. AdamW optimizer, cosine LR schedule.
    #   5. For each epoch:
    #         - Train one epoch with vlm.clip.clip_loss.
    #         - Clamp logit_scale.data to <= ln(100).
    #         - Compute zero-shot val accuracy via vlm.eval.zeroshot_classification_accuracy.
    #         - Log to stdout (and W&B if args.wandb).
    #   6. Save the best checkpoint to args.output_dir / "best.pt".
    img_size = cfg["vit"]["img_size"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]
    patch_size = cfg["vit"]["patch_size"]
    d_model = cfg["vit"]["d_model"]
    num_heads = cfg["vit"]["num_heads"]
    num_blocks = cfg["vit"]["num_blocks"]
    dropout = cfg["vit"]["dropout"]
    lr = cfg["optim"]["lr"]
    weight_decay = cfg["optim"]["weight_decay"]
    betas = cfg["optim"]["betas"]
    warmup_steps = cfg["optim"]["warmup_steps"]
    scheduler = cfg["optim"]["scheduler"]
    num_epochs = cfg["train"]["num_epochs"]
    d_proj = cfg["projection"]["d_proj"]
    device = args.device

    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    vit.to(device)

    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"])
    text_encoder.to(device)
    text_encoder.eval()

    projection_heads = ProjectionHeads(
        d_image=d_model,
        d_text=text_encoder.embedding_dim,
        d_proj=d_proj,
    )
    projection_heads.to(device)

    # Create the logit scale parameter
    logit_scale = init_logit_scale()
    logit_scale.to(device)

    optimizer = AdamW(
        params=chain(vit.parameters(), projection_heads.parameters(), [logit_scale]),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    if scheduler == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=1/warmup_steps, end_factor=1, total_iters=warmup_steps)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
        scheduler = ChainedScheduler([warmup_scheduler, scheduler])
    elif scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")
    
    if args.wandb:
        wandb.init(project="clip-eurosat", config=cfg)

    class_prompts = [f"a satellite image of {class_name}" for class_name in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    for epoch in range(num_epochs):
        train_loss = 0
        for images, captions in tqdm(train_dl, desc="Training", total=len(train_dl)):
            images = images.to(device)
            optimizer.zero_grad()
            image_embeds = vit(images)

            # Encoder uses no_grad internally; ST/HF may still return inference-mode
            # tensors, which MPS cannot retain for Linear backward — clone breaks that.
            text_embeds = text_encoder(captions).clone()

            image_proj, text_proj = projection_heads(image_embeds, text_embeds)

            logit_scale.data.clamp_(max=math.log(100.0))
            loss = clip_loss(image_proj, text_proj, logit_scale)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        mean_train_loss = train_loss / len(train_dl)

        if args.wandb:
            wandb.log({"train_loss": mean_train_loss, "epoch": epoch})

        val_accuracy = zeroshot_classification_accuracy(
            vit,
            projection_heads,
            text_encoder,
            val_dl,
            class_prompts,
            class_indices,
            device,
        )

        if args.wandb:
            wandb.log({"val_accuracy": val_accuracy, "epoch": epoch})
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {mean_train_loss}, Val Accuracy: {val_accuracy}")


if __name__ == "__main__":
    main()
