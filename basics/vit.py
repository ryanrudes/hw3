"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.model import Block

class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # TODO: implement.
        # Hint: use nn.Conv2d with kernel_size=patch_size, stride=patch_size,
        # in_channels=3, out_channels=d_model. Then flatten the spatial dims
        # and transpose so each patch is a token.
        self.conv = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model), unless
         `return_all_tokens=True`, in which case return the full sequence
         (B, num_patches+1, d_model) including CLS.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO: implement.
        # Hint: store self.cls_token as nn.Parameter(torch.zeros(1, 1, d_model))
        # and self.pos_embed as nn.Parameter(torch.zeros(1, num_patches+1, d_model)).
        # Use basics.model.Block(..., is_decoder=False) for the encoder blocks.
        N = (img_size // patch_size) ** 2

        self.d_model = d_model
        self.num_patches = N

        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))
        self.blocks = nn.ModuleList([Block(d_model, num_heads, N + 1, dropout=dropout) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        x = self.patch_embed(x)
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0, :]


if __name__ == "__main__":
    from time import time
    import numpy as np
    
    for patch_size in [8, 16, 32]:
        vit = ViT(img_size=224, patch_size=patch_size, d_model=384, num_heads=6, num_blocks=6)
        vit = vit.to("mps")
        
        # Warmup
        for _ in range(5):
            x = torch.randn(16, 3, 224, 224, device="mps")
            vit(x)

        # Measure time
        x = torch.randn(16, 3, 224, 224, device="mps")
        times = []
        for _ in range(20):
            start = time()
            torch.mps.synchronize()
            vit(x)
            torch.mps.synchronize()
            end = time()
            times.append(end - start)
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"Patch size {patch_size}: {mean_time} ± {std_time} seconds")