"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]

IGNORE_INDEX = -100


def _last_nonpad_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Index the final real token per row (right-padded sequences)."""
    B = logits.size(0)
    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    rows = torch.arange(B, device=logits.device, dtype=torch.long)
    return logits[rows, lengths]


def _expand_labels_prepend(labels: torch.Tensor, num_visual: int) -> torch.Tensor:
    """Shift / pad labels for prepended visual tokens.

    HuggingFace causal LM loss pairs logits[:, i] with labels[:, i+1] (after
    internal padding). Prefix positions 0 .. num_visual-1 with -100 so no
    loss is taken on predicting the next *visual* latent from prior positions.
    """
    if num_visual == 0:
        return labels
    B, T = labels.shape
    out = labels.new_full((B, T + num_visual), IGNORE_INDEX)
    out[:, num_visual:] = labels
    return out


def _apply_padding_to_additive_mask(
    additive: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> torch.Tensor:
    """Zero out attention from/to padded positions on a (B,1,T,T) additive mask."""
    minv = torch.finfo(additive.dtype).min
    valid = attention_mask_2d.to(dtype=additive.dtype).clamp(0.0, 1.0)
    invalid = 1.0 - valid
    # Disallow attending to padded keys (columns).
    additive = additive + invalid.unsqueeze(1).unsqueeze(2) * minv
    # Disallow padded queries from attending anywhere (rows).
    additive = additive + invalid.unsqueeze(1).unsqueeze(3) * minv
    return additive


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) aligned with `input_ids`, or None for inference.
                        Use -100 for positions that must not contribute to the
                        cross-entropy (e.g. padding, question/prompt tokens for
                        answer-only VLM training). Visual-token columns are
                        inserted and masked with -100 inside the forward when
                        labels are provided.
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _encode_visual(self, images: torch.Tensor, return_all_tokens: bool) -> torch.Tensor:
        feats = self.vit(images, return_all_tokens=return_all_tokens)
        x = self.projector(feats)
        dec_dtype = self.decoder.get_input_embeddings().weight.dtype
        return x.to(dec_dtype)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.decoder.get_input_embeddings()
        return emb(input_ids)

    def _decoder_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        mask_mode: MaskMode,
        n_visual: int,
        n_text: int,
    ) -> dict:
        dec_dtype = self.decoder.get_input_embeddings().weight.dtype
        if inputs_embeds.dtype != dec_dtype:
            inputs_embeds = inputs_embeds.to(dec_dtype)
        if mask_mode == "causal":
            out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        elif mask_mode == "image_bidir":
            B, T, _ = inputs_embeds.shape
            if n_visual + n_text != T:
                raise ValueError(
                    f"image_bidir: expected n_visual+n_text==T, got {n_visual}+{n_text}!={T}"
                )
            mask_dtype = inputs_embeds.dtype
            base = build_image_bidir_mask(
                n_visual, n_text, inputs_embeds.device, mask_dtype
            )
            mask_4d = base.expand(B, -1, -1, -1).contiguous()
            mask_4d = _apply_padding_to_additive_mask(mask_4d, attention_mask)
            out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,
                labels=labels,
                use_cache=False,
            )
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")
        loss = getattr(out, "loss", None)
        logits = out.logits
        return {"loss": loss, "logits": logits}

    def _forward_prepend(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        return_all_tokens: bool,
        mask_mode: MaskMode,
    ) -> dict:
        vis = self._encode_visual(images, return_all_tokens=return_all_tokens)
        text_emb = self._embed_tokens(input_ids)
        n_visual = vis.shape[1]
        inputs_embeds = torch.cat([vis, text_emb], dim=1)
        ones = attention_mask.new_ones((attention_mask.shape[0], n_visual))
        full_attn = torch.cat([ones, attention_mask], dim=1)
        adj_labels = _expand_labels_prepend(labels, n_visual) if labels is not None else None
        n_text = input_ids.shape[1]
        return self._decoder_forward(
            inputs_embeds, full_attn, adj_labels, mask_mode, n_visual, n_text
        )

    def _stitch_interleaved_padded(
        self,
        vis: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Replace <image> placeholders with projected patch tokens; right-pad batch."""
        if self.image_token_id is None:
            raise ValueError("interleaved injection requires image_token_id.")
        B, V, _ = vis.shape
        embed_layer = self.decoder.get_input_embeddings()

        seq_embs: list[torch.Tensor] = []
        seq_masks: list[torch.Tensor] = []
        per_row_labels: list[torch.Tensor] = []

        for b in range(B):
            ids = input_ids[b]
            am = attention_mask[b]
            lbl = labels[b] if labels is not None else None
            chunks_e: list[torch.Tensor] = []
            chunks_m: list[torch.Tensor] = []
            chunks_l: list[torch.Tensor] = []
            t = 0
            T = ids.shape[0]
            while t < T:
                if ids[t].item() == self.image_token_id:
                    chunks_e.append(vis[b])
                    chunks_m.append(torch.ones(V, device=am.device, dtype=am.dtype))
                    if lbl is not None:
                        chunks_l.append(
                            ids.new_full((V,), IGNORE_INDEX, dtype=lbl.dtype)
                        )
                    t += 1
                    continue
                t0 = t
                while t < T and ids[t].item() != self.image_token_id:
                    t += 1
                seg_ids = ids[t0:t]
                chunks_e.append(embed_layer(seg_ids))
                chunks_m.append(am[t0:t])
                if lbl is not None:
                    chunks_l.append(lbl[t0:t])
            e = torch.cat(chunks_e, dim=0)
            m = torch.cat(chunks_m, dim=0)
            seq_embs.append(e)
            seq_masks.append(m)
            if lbl is not None:
                per_row_labels.append(torch.cat(chunks_l, dim=0))

        max_len = max(s.shape[0] for s in seq_embs)
        d = seq_embs[0].shape[-1]
        dtype = seq_embs[0].dtype
        pad_emb = torch.zeros(B, max_len, d, device=input_ids.device, dtype=dtype)
        pad_mask = attention_mask.new_zeros((B, max_len))
        pad_labels = (
            labels.new_full((B, max_len), IGNORE_INDEX) if labels is not None else None
        )
        for b in range(B):
            L = seq_embs[b].shape[0]
            pad_emb[b, :L] = seq_embs[b]
            pad_mask[b, :L] = seq_masks[b]
            if pad_labels is not None:
                pad_labels[b, :L] = per_row_labels[b]

        return pad_emb, pad_mask, pad_labels, max_len

    def _forward_interleaved(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        mask_mode: MaskMode,
    ) -> dict:
        if mask_mode == "image_bidir":
            raise ValueError(
                "mask_mode='image_bidir' is only valid when all visual tokens form a "
                "prefix (cls / all_patches). Interleaved text–vision order needs a "
                "different mask; use mask_mode='causal'."
            )
        vis = self._encode_visual(images, return_all_tokens=True)
        pad_emb, pad_mask, pad_labels, max_len = self._stitch_interleaved_padded(
            vis, input_ids, attention_mask, labels
        )
        return self._decoder_forward(
            pad_emb, pad_mask, pad_labels, mask_mode, 0, max_len
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        if injection == "cls":
            return self._forward_prepend(
                images,
                input_ids,
                attention_mask,
                labels,
                return_all_tokens=False,
                mask_mode=mask_mode,
            )
        if injection == "all_patches":
            return self._forward_prepend(
                images,
                input_ids,
                attention_mask,
                labels,
                return_all_tokens=True,
                mask_mode=mask_mode,
            )
        if injection == "interleaved":
            return self._forward_interleaved(
                images, input_ids, attention_mask, labels, mask_mode
            )
        raise ValueError(f"Unknown injection: {injection}")

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        mask_mode: MaskMode = "causal",
        eos_token_id: int | None = None,
        **gen_kwargs,
    ) -> list[str]:
        """Greedy decode an answer continuation given images + prompt strings (no labels).

        `prompts` must already include any ``<image>`` placeholder when using
        ``interleaved`` injection. Decoder and ViT should be in eval mode.
        """
        del gen_kwargs  # reserved for API compatibility
        self.eval()
        device = images.device
        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for generation.")

        return_all = injection in ("all_patches", "interleaved")
        vis = self._encode_visual(images, return_all_tokens=return_all)
        if injection in ("cls", "all_patches"):
            n_visual = vis.shape[1]
            finished = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
            gen_ids: list[list[int]] = [[] for _ in range(images.shape[0])]

            for _ in range(max_new_tokens):
                text_emb = self._embed_tokens(input_ids)
                inputs_embeds = torch.cat([vis, text_emb], dim=1)
                ones = attention_mask.new_ones((attention_mask.shape[0], n_visual))
                full_attn = torch.cat([ones, attention_mask], dim=1)
                n_text = input_ids.shape[1]
                out = self._decoder_forward(
                    inputs_embeds, full_attn, None, mask_mode, n_visual, n_text
                )
                next_logits = _last_nonpad_logits(out["logits"], full_attn)
                next_ids = next_logits.argmax(dim=-1)
                for b in range(images.shape[0]):
                    if finished[b]:
                        continue
                    nid = int(next_ids[b].item())
                    gen_ids[b].append(nid)
                    if nid == eos_token_id:
                        finished[b] = True
                if bool(finished.all().item()):
                    break
                input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((images.shape[0], 1))],
                    dim=1,
                )
            return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # interleaved + causal only (enforced in _forward_interleaved)
        if mask_mode != "causal":
            raise ValueError("interleaved generation requires mask_mode='causal'.")
        finished = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
        gen_ids = [[] for _ in range(images.shape[0])]

        for _ in range(max_new_tokens):
            pad_emb, pad_mask, _, _ = self._stitch_interleaved_padded(
                vis, input_ids, attention_mask, labels=None
            )
            L = pad_emb.shape[1]
            out = self._decoder_forward(pad_emb, pad_mask, None, "causal", 0, L)
            next_logits = _last_nonpad_logits(out["logits"], pad_mask)
            next_ids = next_logits.argmax(dim=-1)
            for b in range(images.shape[0]):
                if finished[b]:
                    continue
                nid = int(next_ids[b].item())
                gen_ids[b].append(nid)
                if nid == eos_token_id:
                    finished[b] = True
            if bool(finished.all().item()):
                break
            input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((images.shape[0], 1))],
                dim=1,
            )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
