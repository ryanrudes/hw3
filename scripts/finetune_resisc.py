"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt

LoRA rank sweep (α = k·r with fixed k from config, default k=2):
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --pretrained runs/clip_eurosat/best.pt --lora-rank-sweep

Same sweep, logged as a Weights & Biases Sweep (grid over ``rank``):
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --pretrained runs/clip_eurosat/best.pt --lora-rank-sweep --wandb-sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from tqdm import tqdm

from basics.lora import apply_lora_to_attention
from basics.vit import ViT
from vlm.data import build_resisc45_loaders

# Must match the ViT used in §3 CLIP pretraining (`configs/clip_eurosat.yaml`).
_DEFAULT_VIT = {
    "img_size": 64,
    "patch_size": 8,
    "d_model": 384,
    "num_heads": 6,
    "num_blocks": 6,
    "dropout": 0.1,
}

_DEFAULT_LORA_SWEEP_RANKS = [1, 2, 4, 8, 16, 32, 64]


class ViTClassifier(nn.Module):
    def __init__(self, vit: ViT, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument(
        "--lora-rank-sweep",
        action="store_true",
        help="Run LoRA on ranks from config (default 1..64 powers of two) with α=k·r; "
        "writes per-rank metrics, sweep_summary.json, and lora_rank_vs_accuracy.png.",
    )
    p.add_argument(
        "--wandb-sweep",
        action="store_true",
        help="Requires --lora-rank-sweep. Registers a W&B Sweep (grid over rank) and runs "
        "wandb.agent locally so trials attach to one sweep.",
    )
    return p.parse_args()


def _accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def _lora_sweep_settings(cfg: dict) -> tuple[list[int], float]:
    block = cfg.get("lora_rank_sweep") or {}
    ranks = list(block.get("ranks", _DEFAULT_LORA_SWEEP_RANKS))
    alpha_scale = float(block.get("alpha_scale", 2.0))
    return ranks, alpha_scale


def _plot_lora_rank_sweep(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda d: d["rank"])
    ranks = [int(d["rank"]) for d in rows]
    accs = [float(d["test_accuracy"]) for d in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(ranks, accs, marker="o", color="tab:blue")
    ax.set_xticks(ranks)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Final test accuracy")
    ax.set_title(r"RESISC45: test accuracy vs LoRA rank ($\alpha = k r$, fixed $k$)")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    fig.tight_layout()
    out = output_dir / "lora_rank_vs_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def run_resisc_finetune(
    cfg: dict,
    *,
    device: torch.device,
    pretrained: Path,
    method: str,
    rank: int,
    alpha: float,
    output_dir: Path,
    use_wandb: bool,
    wandb_run_name: str | None = None,
    external_wandb: bool = False,
) -> dict[str, Any]:
    """One full train + eval; writes ``output_dir / metrics.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vit_cfg = {**_DEFAULT_VIT, **cfg.get("vit", {})}
    num_classes = int(cfg["num_classes"])
    method_optim = cfg["methods"][method]
    lr = float(method_optim.get("lr", cfg["optim"]["lr"]))
    weight_decay = float(cfg["optim"]["weight_decay"])
    betas = tuple(cfg["optim"]["betas"])
    batch_size = int(cfg["train"]["batch_size"])
    num_epochs = int(cfg["train"]["num_epochs"])
    num_workers = int(cfg["train"]["num_workers"])
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every_epoch"])

    train_dl, test_dl = build_resisc45_loaders(
        img_size=int(vit_cfg["img_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    vit = ViT(
        img_size=int(vit_cfg["img_size"]),
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg["dropout"]),
    )
    ckpt = torch.load(pretrained, map_location=device, weights_only=False)
    vit.load_state_dict(ckpt["vit_state_dict"])
    vit.to(device)

    if method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad = False
        model = ViTClassifier(vit, num_classes).to(device)
    elif method == "lora":
        apply_lora_to_attention(vit, rank, alpha)
        model = ViTClassifier(vit, num_classes).to(device)
    else:
        model = ViTClassifier(vit, num_classes).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    num_trainable = sum(p.numel() for p in trainable_params)

    total_steps = num_epochs * len(train_dl)
    warmup_steps_cfg = int(cfg["optim"]["warmup_steps"])
    warmup_steps = min(warmup_steps_cfg, total_steps)
    sched_name = cfg["optim"].get("scheduler")
    if sched_name == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1 / max(1, warmup_steps_cfg),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        scheduler: ChainedScheduler | None = ChainedScheduler([warmup_scheduler, cosine_scheduler])
    elif sched_name is None:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name!r}")

    mem_device: str | None = str(device) if str(device).startswith("cuda") else None
    if mem_device is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(mem_device)

    wall_t0 = time.perf_counter()

    if use_wandb and not external_wandb:
        init_kwargs: dict[str, Any] = {
            "project": "resisc45-finetune",
            "config": {
                **cfg,
                "method": method,
                "rank": rank,
                "alpha": alpha,
                "pretrained": str(pretrained),
                "vit": vit_cfg,
            },
        }
        if wandb_run_name:
            init_kwargs["name"] = wandb_run_name
        wandb.init(**init_kwargs)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{num_epochs}", total=len(train_dl))
        for batch_i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            n_batches += 1

            if log_every > 0 and (batch_i + 1) % log_every == 0:
                avg = running_loss / n_batches
                pbar.set_postfix(loss=f"{avg:.4f}")

        mean_train_loss = running_loss / max(n_batches, 1)

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if (epoch + 1) % eval_every == 0:
            test_acc = _accuracy(model, test_dl, device)
            print(f"epoch {epoch + 1}: train_loss={mean_train_loss:.4f} test_acc={test_acc:.4f}")
            if use_wandb:
                wandb.log({"epoch": epoch, "test_accuracy": test_acc})

    wall_sec = time.perf_counter() - wall_t0
    if mem_device is not None and torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated(mem_device))
    else:
        peak_bytes = 0

    final_test_acc = _accuracy(model, test_dl, device)
    metrics: dict[str, Any] = {
        "test_accuracy": float(final_test_acc),
        "num_trainable_params": int(num_trainable),
        "peak_memory_bytes": peak_bytes,
        "wall_clock_sec": float(wall_sec),
        "method": method,
    }
    if method == "lora":
        metrics["rank"] = rank
        metrics["alpha"] = float(alpha)
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")
    print(metrics)

    if use_wandb:
        wandb.log(metrics)
    if use_wandb and not external_wandb:
        wandb.finish()

    return metrics


def _wandb_sweep_project(cfg: dict) -> str:
    block = cfg.get("lora_rank_sweep") or {}
    return str(block.get("wandb_project", "resisc45-finetune"))


def _wandb_sweep_name(cfg: dict) -> str:
    block = cfg.get("lora_rank_sweep") or {}
    return str(block.get("wandb_sweep_name", "lora-rank-resisc45"))


def _execute_wandb_lora_rank_sweep(
    cfg: dict,
    *,
    args: argparse.Namespace,
    device: torch.device,
    base_out: Path,
    ranks: list[int],
    alpha_scale: float,
) -> None:
    """Register a W&B grid sweep over ``rank`` and run all trials via ``wandb.agent``."""
    project = _wandb_sweep_project(cfg)
    sweep_name = _wandb_sweep_name(cfg)
    sweep_config: dict[str, Any] = {
        "name": sweep_name,
        "method": "grid",
        "metric": {"name": "test_accuracy", "goal": "maximize"},
        "parameters": {"rank": {"values": [int(r) for r in ranks]}},
    }
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"W&B sweep id: {sweep_id} (project={project})")

    summary_holder: list[dict[str, Any]] = []

    def _sweep_train() -> None:
        wandb.init(
            project=project,
            group=sweep_name,
            job_type="lora-rank-trial",
            config={
                "alpha_scale": alpha_scale,
                "pretrained": str(args.pretrained),
                "config_file": str(args.config.resolve()),
            },
        )
        r = int(wandb.config.rank)
        alpha = alpha_scale * r
        rank_dir = base_out / f"rank_{r}"
        print(f"\n=== W&B sweep trial rank={r} alpha={alpha} (alpha/r={alpha_scale}) ===\n")
        m = run_resisc_finetune(
            cfg,
            device=device,
            pretrained=args.pretrained,
            method="lora",
            rank=r,
            alpha=float(alpha),
            output_dir=rank_dir,
            use_wandb=True,
            wandb_run_name=None,
            external_wandb=True,
        )
        summary_holder.append(dict(m))
        wandb.finish()

    wandb.agent(sweep_id, function=_sweep_train, count=len(ranks))

    summary: list[dict[str, Any]] = []
    for r in ranks:
        p = base_out / f"rank_{r}" / "metrics.json"
        if p.is_file():
            with open(p) as f:
                summary.append(json.load(f))
    if not summary:
        summary = summary_holder
    summary = sorted(summary, key=lambda d: int(d["rank"]))
    sweep_path = base_out / "sweep_summary.json"
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {sweep_path}")
    _plot_lora_rank_sweep(base_out, summary)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    if args.wandb_sweep and not args.lora_rank_sweep:
        raise SystemExit("--wandb-sweep requires --lora-rank-sweep")

    if args.lora_rank_sweep:
        if args.method != "lora":
            raise SystemExit("--lora-rank-sweep requires --method lora")
        ranks, alpha_scale = _lora_sweep_settings(cfg)
        base_out = args.output_dir or Path("runs") / "resisc_lora_rank_sweep"
        base_out.mkdir(parents=True, exist_ok=True)

        if args.wandb_sweep:
            if args.wandb:
                print("Note: --wandb-sweep owns W&B logging; ignoring --wandb.")
            _execute_wandb_lora_rank_sweep(
                cfg,
                args=args,
                device=device,
                base_out=base_out,
                ranks=ranks,
                alpha_scale=alpha_scale,
            )
            return

        summary: list[dict[str, Any]] = []
        for r in ranks:
            alpha = alpha_scale * r
            rank_dir = base_out / f"rank_{r}"
            wname = f"lora-r{r}-a{alpha:g}" if args.wandb else None
            print(f"\n=== LoRA sweep rank={r} alpha={alpha} (alpha/r={alpha_scale}) ===\n")
            m = run_resisc_finetune(
                cfg,
                device=device,
                pretrained=args.pretrained,
                method="lora",
                rank=int(r),
                alpha=float(alpha),
                output_dir=rank_dir,
                use_wandb=args.wandb,
                wandb_run_name=wname,
                external_wandb=False,
            )
            summary.append(dict(m))

        sweep_path = base_out / "sweep_summary.json"
        with open(sweep_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {sweep_path}")
        _plot_lora_rank_sweep(base_out, summary)
        return

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    run_resisc_finetune(
        cfg,
        device=device,
        pretrained=args.pretrained,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_run_name=None,
    )


if __name__ == "__main__":
    main()
