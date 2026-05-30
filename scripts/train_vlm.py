"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt --injection all_patches --mask-mode image_bidir

``--pretrained-vit`` must be the EuroSAT CLIP checkpoint from §3 (``pretrain_clip.py`` with
``configs/clip_eurosat.yaml``), i.e. ``best.pt`` containing ``vit_state_dict`` — not a randomly
initialized ViT.

Run all three injection strategies (2000 steps each, projector-only) and print a summary table:

    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt --all-injections

Log to Weights & Biases (optional):

    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt --injection all_patches --wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from itertools import cycle
from pathlib import Path

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.vit import ViT
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy
from vlm.model import InjectionMode, MaskMode, VisionLanguageModel
from vlm.projector import VisionLanguageProjector


def _ensure_smoke_assets(cfg: dict, args: argparse.Namespace) -> None:
    """Minimal CLEVR-style layout + random ViT checkpoint for ``--smoke``."""
    root = Path(cfg["data"]["root"])
    (root / "images").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    png = root / "images" / "000000.png"
    if not png.exists():
        Image.new("RGB", (64, 64), color=(90, 120, 60)).save(png)
    for split, n in (("train", 48), ("val", 12)):
        jp = root / f"{split}.jsonl"
        if jp.exists():
            continue
        with open(jp, "w") as f:
            for i in range(n):
                rec = {
                    "image_file": "000000.png",
                    "question": f"How many objects (sample {i})?",
                    "answer": str(i % 5),
                    "q_type": "other",
                }
                f.write(json.dumps(rec) + "\n")
    if not args.pretrained_vit.exists():
        vcfg = cfg["vit"]
        vit = ViT(
            img_size=vcfg["img_size"],
            patch_size=vcfg["patch_size"],
            d_model=vcfg["d_model"],
            num_heads=vcfg["num_heads"],
            num_blocks=vcfg["num_blocks"],
            dropout=0.0,
        )
        out = Path("runs/_smoke_vit_init.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vit_state_dict": vit.state_dict()}, out)
        args.pretrained_vit = out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--pretrained-vit",
        type=Path,
        required=True,
        help=(
            "EuroSAT CLIP ViT from §3: best.pt from pretrain_clip.py (clip_eurosat.yaml), "
            "with vit_state_dict. Required for real §5 runs; --smoke may synthesize a dummy ViT."
        ),
    )
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="A=projector only (required for the §5 injection sweep).",
    )
    p.add_argument(
        "--all-injections",
        action="store_true",
        help="Train cls, all_patches, and interleaved sequentially and emit a comparison table.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny dry run (2 steps, 8 train examples) for debugging.",
    )
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument(
        "--wandb-project",
        default="vlm-clevr",
        help="W&B project name (default: vlm-clevr).",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name; default is vlm_{injection}_{mask_mode}.",
    )
    return p.parse_args()


def visual_token_count(img_size: int, patch_size: int, injection: str) -> int:
    if injection == "cls":
        return 1
    n = (img_size // patch_size) ** 2
    return n + 1  # CLS + patches


def effective_mask_mode(injection: str, mask_mode: str) -> str:
    if injection == "interleaved" and mask_mode == "image_bidir":
        return "causal"
    return mask_mode


def build_prompt_and_prefix_lens(
    tokenizer,
    questions: list[str],
    answers: list[str],
    injection: str,
) -> tuple[list[str], list[int]]:
    prompts: list[str] = []
    lens: list[int] = []
    for q, a in zip(questions, answers):
        if injection == "interleaved":
            prefix = f"<image>\nQuestion: {q}\nAnswer: "
        else:
            prefix = f"Question: {q}\nAnswer: "
        full = prefix + a
        prompts.append(full)
        pref_ids = tokenizer.encode(prefix, add_special_tokens=True)
        lens.append(len(pref_ids))
    return prompts, lens


def tokenize_batch(tokenizer, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def make_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_lens: list[int],
    pad_id: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    for b, pl in enumerate(prefix_lens):
        labels[b, :pl] = -100
        labels[b][attention_mask[b] == 0] = -100
    return labels


def load_vit(cfg: dict, ckpt_path: Path, device: str) -> ViT:
    vcfg = cfg["vit"]
    vit = ViT(
        img_size=vcfg["img_size"],
        patch_size=vcfg["patch_size"],
        d_model=vcfg["d_model"],
        num_heads=vcfg["num_heads"],
        num_blocks=vcfg["num_blocks"],
        dropout=vcfg["dropout"],
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["vit_state_dict"] if isinstance(blob, dict) and "vit_state_dict" in blob else blob
    vit.load_state_dict(sd, strict=True)
    vit.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    return vit


def load_decoder_tokenizer(cfg: dict, device: str, injection: str):
    dcfg = cfg["decoder"]
    model_name = dcfg["model_name"]
    td = dcfg["torch_dtype"]
    torch_dtype = getattr(torch, td) if isinstance(td, str) else td
    attn_impl = dcfg.get("attn_implementation", "sdpa")
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            use_safetensors=True,
        )
    except (ImportError, ValueError, TypeError, RuntimeError, OSError):
        try:
            decoder = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
                use_safetensors=True,
            )
        except OSError:
            decoder = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    image_token_id: int | None = None
    if injection == "interleaved":
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image>"]}
        )
        if n_added == 0 and tokenizer.convert_tokens_to_ids("<image>") == tokenizer.unk_token_id:
            raise RuntimeError("Failed to add <image> token to tokenizer.")
        decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    decoder.to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder, tokenizer, image_token_id


@torch.no_grad()
def evaluate_exact_match(
    model: VisionLanguageModel,
    val_subset_dl: DataLoader,
    device: torch.device,
    max_examples: int,
    injection: InjectionMode,
    mask_mode: MaskMode,
    max_new_tokens: int,
) -> float:
    model.eval()
    preds: list[str] = []
    golds: list[str] = []
    n = 0
    for batch in val_subset_dl:
        imgs = batch["image"].to(device)
        qs: list[str] = batch["question"]
        ans: list[str] = batch["answer"]
        for i in range(imgs.shape[0]):
            if n >= max_examples:
                break
            if injection == "interleaved":
                prompt = f"<image>\nQuestion: {qs[i]}\nAnswer: "
            else:
                prompt = f"Question: {qs[i]}\nAnswer: "
            pred = model.generate(
                imgs[i : i + 1],
                [prompt],
                injection=injection,
                mask_mode=mask_mode,
                max_new_tokens=max_new_tokens,
            )[0]
            preds.append(pred.strip())
            golds.append(ans[i].strip())
            n += 1
        if n >= max_examples:
            break
    return float(batch_clevr_accuracy(preds, golds)["overall"])


def train_one_run(
    args: argparse.Namespace,
    cfg: dict,
    injection: InjectionMode,
    requested_mask_mode: MaskMode,
    output_dir: Path,
    wandb_group_id: str | None = None,
) -> dict:
    device = torch.device(args.device)
    mask_mode_eff: MaskMode = effective_mask_mode(injection, requested_mask_mode)  # type: ignore[assignment]

    dcfg = cfg["data"]
    train_dl, val_dl = build_clevr_loaders(
        img_size=dcfg.get("img_size", cfg["vit"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    val_ds = val_dl.dataset
    eval_n = min(cfg["train"]["eval_max_examples"], len(val_ds))
    val_subset = Subset(val_ds, list(range(eval_n)))
    val_eval_dl = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=getattr(val_dl, "collate_fn", None),
    )

    vit = load_vit(cfg, args.pretrained_vit, str(device))
    decoder, tokenizer, image_token_id = load_decoder_tokenizer(cfg, str(device), injection)

    d_dec = decoder.config.hidden_size
    d_img = cfg["vit"]["d_model"]
    pcfg = cfg["projector"]
    projector = VisionLanguageProjector(
        d_image=d_img,
        d_decoder=d_dec,
        expansion=pcfg.get("expansion", 4),
    ).to(device)
    projector.train()

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)
    model.vit.eval()
    model.decoder.eval()
    model.projector.train()

    if args.freeze_config != "A":
        raise ValueError("This sweep expects --freeze-config A (projector only).")

    lr = float(cfg["optim"]["lr"])
    opt_cfg = cfg["optim"]
    optimizer = AdamW(
        projector.parameters(),
        lr=lr,
        betas=tuple(opt_cfg["betas"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )

    num_steps = int(cfg["train"]["num_steps"])
    warmup_steps = min(int(opt_cfg["warmup_steps"]), num_steps)
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_steps, 1),
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(num_steps - warmup_steps, 1),
        eta_min=lr * 0.1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    accum = max(int(cfg["train"].get("gradient_accumulation_steps", 1)), 1)
    max_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    log_every = int(cfg["train"].get("log_every", 25))
    gen_cfg = cfg.get("generation", {})

    use_cuda_bf16 = device.type == "cuda" and str(cfg["decoder"]["torch_dtype"]) == "bfloat16"
    train_it = cycle(train_dl)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.projector.train()
    model.vit.eval()
    model.decoder.eval()

    if args.wandb:
        run_name = args.wandb_run_name or f"vlm_{injection}_{mask_mode_eff}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=wandb_group_id,
            config=cfg,
            reinit=True,
        )
        wandb.config.update(
            {
                "injection": injection,
                "mask_mode": mask_mode_eff,
                "freeze_config": args.freeze_config,
                "pretrained_vit": str(args.pretrained_vit),
                "output_dir": str(output_dir),
                "smoke": args.smoke,
            },
            allow_val_change=True,
        )

    step_times: list[float] = []
    global_step = 0
    pbar = tqdm(total=num_steps, desc=f"train[{injection}]")
    loss_ema = None

    try:
        while global_step < num_steps:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                batch = next(train_it)
                imgs = batch["image"].to(device, non_blocking=True)
                qs: list[str] = batch["question"]
                ans: list[str] = batch["answer"]
                texts, prefix_lens = build_prompt_and_prefix_lens(
                    tokenizer, qs, ans, injection
                )
                enc = tokenize_batch(tokenizer, texts, device)
                input_ids = enc["input_ids"]
                attn = enc["attention_mask"]
                labels = make_labels(
                    input_ids,
                    attn,
                    prefix_lens,
                    tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda_bf16
                ):
                    out = model(
                        imgs,
                        input_ids,
                        attn,
                        labels=labels,
                        injection=injection,
                        mask_mode=mask_mode_eff,
                    )
                    loss = out["loss"] / accum
                loss.backward()

            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            step_times.append(time.perf_counter() - t0)
            global_step += 1
            pbar.update(1)

            if loss_ema is None:
                loss_ema = float(loss.detach().item() * accum)
            else:
                loss_ema = 0.98 * loss_ema + 0.02 * float(loss.detach().item() * accum)

            if global_step % log_every == 0:
                pbar.set_postfix(loss=f"{loss_ema:.4f}")
                if args.wandb:
                    wandb.log(
                        {
                            "train/loss_ema": loss_ema,
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

        pbar.close()

        peak_gb = None
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        acc = evaluate_exact_match(
            model,
            val_eval_dl,
            device,
            eval_n,
            injection,
            mask_mode_eff,
            int(gen_cfg.get("max_new_tokens", 32)),
        )

        n_vis = visual_token_count(cfg["vit"]["img_size"], cfg["vit"]["patch_size"], injection)
        mean_step_s = sum(step_times) / max(len(step_times), 1)
        summary = {
            "injection": injection,
            "mask_mode": mask_mode_eff,
            "val_exact_match": acc,
            "visual_tokens_per_example": n_vis,
            "peak_gpu_memory_gb": peak_gb,
            "mean_wall_time_per_step_s": mean_step_s,
            "num_steps": num_steps,
            "batch_size": cfg["train"]["batch_size"],
            "lr": lr,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        torch.save({"projector": projector.state_dict()}, output_dir / "projector.pt")

        if args.wandb:
            log_final: dict[str, float | int | str | None] = {
                "val/exact_match": acc,
                "val/eval_max_examples": eval_n,
                "model/visual_tokens_per_example": n_vis,
                "train/mean_wall_time_per_step_s": mean_step_s,
                "train/num_steps": num_steps,
            }
            if peak_gb is not None:
                log_final["system/peak_gpu_memory_gb"] = peak_gb
            wandb.log(log_final, step=num_steps)

        return summary
    finally:
        if args.wandb and wandb.run is not None:
            wandb.finish()


def print_markdown_table(rows: list[dict]) -> None:
    print()
    print("| Injection | Val exact-match (500 ex.) | Visual tokens | Peak GPU (GB) | s / step |")
    print("|-----------|---------------------------|---------------|---------------|----------|")
    for r in rows:
        peak = r.get("peak_gpu_memory_gb")
        peak_s = f"{peak:.2f}" if peak is not None else "n/a (CPU)"
        print(
            f"| {r['injection']} | {r['val_exact_match']:.4f} | "
            f"{int(r['visual_tokens_per_example'])} | {peak_s} | {r['mean_wall_time_per_step_s']:.4f} |"
        )
    print()


def discussion_paragraph(rows: list[dict]) -> str:
    cls_r = next(x for x in rows if x["injection"] == "cls")
    patches_r = next(x for x in rows if x["injection"] == "all_patches")
    inter_r = next(x for x in rows if x["injection"] == "interleaved")
    acc_max = max(r["val_exact_match"] for r in rows)
    winners = [r["injection"] for r in rows if r["val_exact_match"] == acc_max]

    def oxford(names: list[str]) -> str:
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    if len(winners) == 1:
        lead = (
            f"The best validation exact-match was **{winners[0]}** ({acc_max:.1%}). "
        )
    else:
        lead = f"{oxford(winners)} tied for the best validation exact-match ({acc_max:.1%}). "

    mem_p = float(patches_r.get("peak_gpu_memory_gb") or 0.0)
    mem_c = float(cls_r.get("peak_gpu_memory_gb") or 0.0)
    t_p = float(patches_r["mean_wall_time_per_step_s"])
    t_c = float(cls_r["mean_wall_time_per_step_s"])
    t_i = float(inter_r["mean_wall_time_per_step_s"])

    return (
        lead
        + "This connects directly to the CLS-vs-patch pooling question from the ViT pooling problem: "
        "CLS compresses the whole image into one token, while **all_patches** and **interleaved** "
        f"both expose the LM to the same {int(patches_r['visual_tokens_per_example'])} visual tokens "
        "(CLS plus every patch), giving the decoder spatial structure instead of a single pooled vector. "
        "That extra context usually improves fine-grained reasoning when the projector and LM can exploit it, "
        f"at the cost of longer sequences: here peak GPU memory was about {mem_p:.2f} GB for **all_patches** "
        f"versus {mem_c:.2f} GB for **cls**, and mean wall-clock time per optimizer step was "
        f"{t_c:.4f} s (cls), {t_p:.4f} s (all_patches), and {t_i:.4f} s (interleaved). "
        "Whether the extra cost is worth it is answered by the accuracy gap: a large gain favors patch-style "
        "injection; a tiny gain favors the cheaper single-token design."
    )


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f)

    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["decoder"] = {
            "model_name": "openai-community/gpt2",
            "torch_dtype": "float32",
            "attn_implementation": "sdpa",
        }
        cfg["train"]["num_steps"] = 2
        cfg["train"]["batch_size"] = min(4, int(cfg["train"]["batch_size"]))
        cfg["train"]["eval_max_examples"] = 8
        cfg["train"]["gradient_accumulation_steps"] = 1
        _ensure_smoke_assets(cfg, args)
    elif not args.pretrained_vit.exists():
        raise FileNotFoundError(
            f"ViT checkpoint not found: {args.pretrained_vit}. "
            "Pass a valid --pretrained-vit path (e.g. CLIP run best.pt), or use --smoke."
        )

    wandb_group_id: str | None = None
    if args.wandb and args.all_injections:
        wandb_group_id = f"vlm_all_injections_{uuid.uuid4().hex[:10]}"

    if args.all_injections:
        runs: list[dict] = []
        for inj, mmode in (
            ("cls", "image_bidir"),
            ("all_patches", "image_bidir"),
            ("interleaved", "causal"),
        ):
            out = (
                Path("runs")
                / f"vlm_clevr_{inj}_{effective_mask_mode(inj, mmode)}_A"
                if args.output_dir is None
                else args.output_dir / f"{inj}"
            )
            row = train_one_run(
                args,
                cfg,
                inj,  # type: ignore[arg-type]
                mmode,  # type: ignore[arg-type]
                out,
                wandb_group_id=wandb_group_id,
            )
            runs.append(row)
        print_markdown_table(runs)
        print(discussion_paragraph(runs))
        sweep_path = (
            args.output_dir / "vlm_clevr_injection_sweep.json"
            if args.output_dir is not None
            else Path("runs") / "vlm_clevr_injection_sweep.json"
        )
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(runs, f, indent=2)
        return

    if args.output_dir is None:
        mm = effective_mask_mode(args.injection, args.mask_mode)
        args.output_dir = Path("runs") / f"vlm_{args.injection}_{mm}_{args.freeze_config}"
    summary = train_one_run(
        args,
        cfg,
        args.injection,  # type: ignore[arg-type]
        args.mask_mode,  # type: ignore[arg-type]
        args.output_dir,
        wandb_group_id=None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
