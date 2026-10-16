# Adapted from EoMT, copyright 2025 Mobile Perception Systems Lab, TU/e (MIT).
# Portions of the attention implementation originate from timm (Apache-2.0).
"""DINOv3 ViT-L/16 segmenter: 10 RGB blocks + 10 fused blocks + 4 query blocks.

Parameter names deliberately match the original RiSE checkpoints. Inputs are
RGB tensors in [0, 255]; normalization happens exactly once in forward().
"""

import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from .experts import apply_experts, set_expert
from .scale_block import ScaleBlock
from .dct import DCTEncoder


class RiSE(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.config = config
        self.finetune_mode = "pdfe2ffort"
        # Normal Hugging Face login / HF_TOKEN handles access; no embedded tokens.
        if initialize:
            backbone = AutoModel.from_pretrained(config.backbone)
        else:
            backbone = AutoModel.from_config(
                AutoConfig.from_pretrained(config.backbone)
            )
        self.backbone = self.transformers_to_timm(backbone, (512, 512))
        if self.backbone.embed_dim != 1024 or len(self.backbone.blocks) != 24:
            raise ValueError(
                "RiSE release configs require DINOv3 ViT-L/16: 1024 channels and 24 blocks"
            )
        apply_experts(
            self.backbone, config.num_experts, config.residual_rank, initialize
        )
        self.num_q = config.num_queries
        self.num_forg_type = config.num_experts
        self.num_classes = 1
        self.num_blocks = 4
        self.query_mode = config.query_mode
        self.head_mode = "single"
        self.qh_mode = self.query_mode
        self.use_multi_query = config.query_mode == "multi"
        self.use_multi_head = self.use_multi_qh = False
        self.use_dct_feature = config.use_dct
        self.require_dct_input_when_enabled = config.use_dct
        self.dct_fuse_block_idx = config.dct_fuse_block_idx
        self.dct_fuse_prob = config.dct_fuse_prob
        dim = self.backbone.embed_dim
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))
        self.dct_encoder = (
            DCTEncoder(out_dim=dim, patch_size=16) if config.use_dct else None
        )
        count = self.num_q * (self.num_forg_type if self.use_multi_query else 1)
        self.q_table = nn.Embedding(count, dim)
        self.class_head = nn.Linear(dim, 2)
        self.mask_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.upscale = nn.Sequential(ScaleBlock(dim), ScaleBlock(dim))
        self.class_heads = self.mask_heads = self.upscale_heads = None

    def _norm_forg_ids(
        self,
        forg_type_ids: Optional[torch.LongTensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if forg_type_ids is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        forg_type_ids = forg_type_ids.to(device=device, dtype=torch.long).view(-1)
        if forg_type_ids.numel() != batch_size:
            raise ValueError(
                f"forg_type_ids must have shape ({batch_size},), got {tuple(forg_type_ids.shape)}"
            )
        if torch.any((forg_type_ids < 0) | (forg_type_ids >= self.num_forg_type)):
            raise ValueError(
                f"forg_type_ids values must be in [0, {self.num_forg_type - 1}]"
            )
        return forg_type_ids

    def _select_expert_queries(self, forg_type_ids: torch.LongTensor) -> torch.Tensor:
        forg_type_ids = forg_type_ids.to(dtype=torch.long)
        if self.use_multi_query:
            base = forg_type_ids.unsqueeze(1) * self.num_q
            offsets = torch.arange(self.num_q, device=forg_type_ids.device).unsqueeze(0)
            query_ids = base + offsets
        else:
            query_ids = (
                torch.arange(self.num_q, device=forg_type_ids.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(forg_type_ids.shape[0], -1)
            )
        return self.q_table(query_ids)

    def apply_class_head(self, q, forg_type_ids=None):
        return self.class_head(q)

    def apply_mask_head(self, q, forg_type_ids=None):
        return self.mask_head(q)

    def apply_upscale(self, features, forg_type_ids=None):
        return self.upscale(features)

    def transformers_to_timm(self, backbone, img_size):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )
        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        if hasattr(backbone, "layer"):
            backbone.blocks = backbone.layer
        elif hasattr(backbone, "model") and hasattr(backbone.model, "layer"):
            backbone.blocks = backbone.model.layer
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            backbone.blocks = backbone.encoder.layer
        else:
            raise AttributeError(
                f"Unsupported backbone structure for {backbone.__class__.__name__}: cannot find transformer layers at .layer, .model.layer, or .encoder.layer."
            )
        if hasattr(backbone.patch_embed, "mask_token"):
            del backbone.patch_embed.mask_token
        del backbone.embeddings
        if hasattr(backbone, "layer"):
            del backbone.layer
        if hasattr(backbone, "model"):
            del backbone.model
        if hasattr(backbone, "encoder"):
            del backbone.encoder
        return backbone

    def _predict(
        self,
        x: torch.Tensor,
        qn: Optional[int] = None,
        forg_type_ids: Optional[torch.LongTensor] = None,
        return_features: bool = False,
    ):
        q = x[:, :qn, :]
        class_logits = self.apply_class_head(q, forg_type_ids=forg_type_ids)
        x = x[:, qn + self.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.backbone.patch_embed.grid_size
        )
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw",
            self.apply_mask_head(q, forg_type_ids=forg_type_ids),
            self.apply_upscale(x, forg_type_ids=forg_type_ids),
        )
        if return_features:
            return (mask_logits, class_logits, q, x)
        return (mask_logits, class_logits, q)

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob, qn: Optional[int] = None):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], qn, device=attn_mask.device) > prob
            )
            attn_mask[:, :qn, qn + self.backbone.num_prefix_tokens :][
                random_queries
            ] = True
        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = (module.q_norm(q), module.k_norm(k))
        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1).contiguous()
        dropout_p = module.attn_drop.p if self.training else 0.0
        if module.fused_attn and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = q @ k.transpose(-2, -1) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v
        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x

    def _attn_mask(
        self,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        i: int,
        qn: Optional[int] = None,
    ):
        B, N = (x.shape[0], x.shape[1])
        attn_mask = torch.ones(B, N, N, dtype=torch.bool, device=x.device)
        interpolated = F.interpolate(
            mask_logits, self.backbone.patch_embed.grid_size, mode="bilinear"
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        patch_start = qn + self.backbone.num_prefix_tokens
        attn_mask[:, :qn, patch_start:] = interpolated > 0
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[i - len(self.backbone.blocks) + self.num_blocks],
            qn=qn,
        )
        return attn_mask

    # Fixed image-level expert ids are training labels; LPC supplies them at inference.
    def forward(
        self,
        x: torch.Tensor,
        forg_type_ids: Optional[torch.LongTensor] = None,
        dct: Optional[torch.Tensor] = None,
        qt: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        x = x / 255.0
        x = (x - self.pixel_mean) / self.pixel_std
        rope = None
        if hasattr(self.backbone, "rope_embeddings"):
            rope = self.backbone.rope_embeddings(x)
        x = self.backbone.patch_embed(x)
        dct_tokens = None
        if self.use_dct_feature:
            if dct is None or qt is None:
                if self.require_dct_input_when_enabled:
                    raise ValueError(
                        "DCT feature is enabled but dct/qt input is missing. Provide both dct and qt for this DCT model."
                    )
            else:
                dct_tokens = self.dct_encoder(dct, qt)
        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        forg_type_ids = self._norm_forg_ids(forg_type_ids, x.shape[0], x.device)
        set_expert(self.backbone, forg_type_ids)
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = ([], [])
        q_sel = self._select_expert_queries(forg_type_ids)
        qn = q_sel.size(1)
        query_concat_idx = len(self.backbone.blocks) - self.num_blocks
        for i, block in enumerate(self.backbone.blocks):
            if (
                dct_tokens is not None
                and i == self.dct_fuse_block_idx
                and (i < query_concat_idx)
            ):
                prefix_n = self.backbone.num_prefix_tokens
                if x.shape[1] != prefix_n + dct_tokens.shape[1]:
                    raise ValueError(
                        f"Unexpected token shape for pre-query DCT fusion: x={tuple(x.shape)}, dct_tokens={tuple(dct_tokens.shape)}"
                    )
                fuse_vec = (
                    torch.rand(x.shape[0], 1, 1, device=x.device) < self.dct_fuse_prob
                ).to(dtype=x.dtype)
                x[:, prefix_n:, :] = x[:, prefix_n:, :] + fuse_vec * dct_tokens.to(
                    dtype=x.dtype
                )
            if i == len(self.backbone.blocks) - self.num_blocks:
                x = torch.cat((q_sel, x), dim=1)
            if i >= len(self.backbone.blocks) - self.num_blocks:
                mask_logits, class_logits, _ = self._predict(
                    self.backbone.norm(x), qn=qn, forg_type_ids=forg_type_ids
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                attn_mask = self._attn_mask(x, mask_logits, i, qn=qn)
            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)
            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)
        if return_features:
            mask_logits, class_logits, query, feat_map = self._predict(
                self.backbone.norm(x),
                qn=qn,
                forg_type_ids=forg_type_ids,
                return_features=True,
            )
        else:
            mask_logits, class_logits, query = self._predict(
                self.backbone.norm(x), qn=qn, forg_type_ids=forg_type_ids
            )
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        if return_features:
            return (mask_logits_per_layer, class_logits_per_layer, query, feat_map)
        return (mask_logits_per_layer, class_logits_per_layer, query)
