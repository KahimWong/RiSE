"""Frozen principal SVD weights plus independently trained residual experts."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualExpertLinear(nn.Module):
    def __init__(self, source, num_experts, rank, initialize=True):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.num_experts = num_experts
        self.weight_main = nn.Parameter(
            torch.empty_like(source.weight), requires_grad=False
        )
        self.bias = (
            nn.Parameter(source.bias.detach().clone())
            if source.bias is not None
            else None
        )
        n = min(source.weight.shape)
        if not 0 < rank < n:
            raise ValueError(f"Residual rank must be in [1, {n-1}]")
        self.expert_U = nn.Parameter(torch.zeros(num_experts, self.out_features, rank))
        self.expert_S = nn.Parameter(torch.zeros(num_experts, rank))
        self.expert_V = nn.Parameter(torch.zeros(num_experts, rank, self.in_features))
        self._batch_expert_idx = None
        if initialize:
            # Only fresh training needs this SVD. Full checkpoints overwrite every tensor.
            u, s, vh = torch.linalg.svd(
                source.weight.detach().float(), full_matrices=False
            )
            principal = n - rank
            self.weight_main.data.copy_(
                (u[:, :principal] * s[:principal]) @ vh[:principal]
            )
            self.expert_U.data.copy_(
                u[:, principal:].unsqueeze(0).expand_as(self.expert_U)
            )
            self.expert_S.data.copy_(
                s[principal:].unsqueeze(0).expand_as(self.expert_S)
            )
            self.expert_V.data.copy_(
                vh[principal:].unsqueeze(0).expand_as(self.expert_V)
            )

    def forward(self, x):
        ids = self._batch_expert_idx
        if ids is None or ids.numel() != x.shape[0]:
            raise ValueError(
                "Set one valid expert id per image before the forward pass"
            )
        # The same expert handles every token of an image. No learned token router.
        v = self.expert_V[ids] * self.expert_S[ids].unsqueeze(-1)
        residual = torch.einsum(
            "bni,bri->bnr", x.reshape(x.shape[0], -1, self.in_features), v
        )
        residual = torch.einsum("bnr,bor->bno", residual, self.expert_U[ids])
        return F.linear(x, self.weight_main, self.bias) + residual.reshape(
            *x.shape[:-1], self.out_features
        )


def set_expert(model, ids):
    for module in model.modules():
        if isinstance(module, ResidualExpertLinear):
            if ids.ndim != 1 or torch.any((ids < 0) | (ids >= module.num_experts)):
                raise ValueError("Expert ids are outside the configured range")
            module._batch_expert_idx = ids


def apply_experts(backbone, num_experts, rank, initialize=True):
    def replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}"
            if isinstance(child, nn.Linear) and any(
                key in path for key in ("attention", "mlp")
            ):
                setattr(
                    parent,
                    name,
                    ResidualExpertLinear(child, num_experts, rank, initialize),
                )
            else:
                replace(child, path)

    replace(backbone)
    # Preserve VFM priors; residual factors and adapted biases remain trainable.
    for p in backbone.parameters():
        p.requires_grad = False
    for module in backbone.modules():
        if isinstance(module, ResidualExpertLinear):
            for p in (module.expert_U, module.expert_S, module.expert_V, module.bias):
                if p is not None:
                    p.requires_grad = True
