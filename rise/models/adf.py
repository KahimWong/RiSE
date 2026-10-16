"""ADF moment propagation adapted from the research implementation."""

import math

import torch
import torch.nn.functional as F


def _gauss_hermite_expectation(func, mu: torch.Tensor, var: torch.Tensor, order: int):
    if order == 3:
        nodes = [-1.224744871391589, 0.0, 1.224744871391589]
        weights = [0.2954089751509193, 1.1816359006036772, 0.2954089751509193]
    elif order == 5:
        nodes = [
            -2.0201828704560856,
            -0.9585724646138185,
            0.0,
            0.9585724646138185,
            2.0201828704560856,
        ]
        weights = [
            0.019953242059045913,
            0.3936193231522412,
            0.9453087204829419,
            0.3936193231522412,
            0.019953242059045913,
        ]
    elif order == 7:
        nodes = [
            -2.6519613568352334,
            -1.6735516287674714,
            -0.8162878828589647,
            0.0,
            0.8162878828589647,
            1.6735516287674714,
            2.6519613568352334,
        ]
        weights = [
            0.0009717812450995191,
            0.05451558281912703,
            0.4256072526101278,
            0.8102646175568073,
            0.4256072526101278,
            0.05451558281912703,
            0.0009717812450995191,
        ]
    else:
        raise ValueError(f"Unsupported Gauss-Hermite order: {order}")

    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var + 1e-8)

    ys = []
    second = []
    norm = 1.0 / math.sqrt(math.pi)
    root2 = math.sqrt(2.0)
    for node, weight in zip(nodes, weights):
        x = mu + root2 * std * node
        y = func(x)
        w = weight * norm
        ys.append(w * y)
        second.append(w * (y**2))

    mu_out = torch.stack(ys, dim=0).sum(dim=0)
    second_moment = torch.stack(second, dim=0).sum(dim=0)
    var_out = torch.clamp(second_moment - mu_out**2, min=0.0)
    return mu_out, var_out


def adf_gelu_approx(mu, var, order=5):
    """Gauss--Hermite GELU moments; order is an explicit inference setting."""
    return _gauss_hermite_expectation(F.gelu, mu, var, order=order)


def adf_layernorm2d(mu: torch.Tensor, var: torch.Tensor, ln_layer):
    weight = ln_layer.weight.view(1, -1, 1, 1)
    bias = ln_layer.bias.view(1, -1, 1, 1)
    stat_mean = mu.mean(dim=1, keepdim=True)
    # Expected LN denominator variance across channels:
    # E[var_c(X)] ≈ var_c(E[X]) + mean_c(Var[X]) * (1 - 1/C)
    # where C is channel count and diagonal-covariance independence is assumed.
    channels = mu.size(1)
    mean_var = var.mean(dim=1, keepdim=True)
    stat_var = mu.var(dim=1, unbiased=False, keepdim=True) + mean_var * (
        1.0 - 1.0 / max(channels, 1)
    )
    scale = weight / torch.sqrt(stat_var + ln_layer.eps)
    mu_out = (mu - stat_mean) * scale + bias

    # Var[X_i - mean_c(X)] = Var[X_i] + Var[mean] - 2Cov(X_i, mean)
    # with diagonal channel assumption: Cov(X_i, mean)=Var[X_i]/C.
    var_mean = var.sum(dim=1, keepdim=True) / (max(channels, 1) ** 2)
    centered_var = var * (1.0 - 2.0 / max(channels, 1)) + var_mean
    var_out = torch.clamp(centered_var * (scale**2), min=0.0)
    return mu_out, var_out


def forward_adf_scaleblock(block, mu: torch.Tensor, var: torch.Tensor, order=5):
    mu = F.conv_transpose2d(
        mu,
        block.conv1.weight,
        block.conv1.bias,
        stride=block.conv1.stride,
        padding=block.conv1.padding,
    )
    var = F.conv_transpose2d(
        var,
        block.conv1.weight**2,
        None,
        stride=block.conv1.stride,
        padding=block.conv1.padding,
    )
    mu, var = adf_gelu_approx(mu, var, order)
    mu = F.conv2d(
        mu,
        block.conv2.weight,
        block.conv2.bias,
        stride=block.conv2.stride,
        padding=block.conv2.padding,
        groups=block.conv2.groups,
    )
    var = F.conv2d(
        var,
        block.conv2.weight**2,
        None,
        stride=block.conv2.stride,
        padding=block.conv2.padding,
        groups=block.conv2.groups,
    )

    mu, var = adf_layernorm2d(mu, var, block.norm)
    return mu, var


def adf_upscale(
    model_ref,
    feat_flat: torch.Tensor,
    noise_var: float,
    forg_type_ids: torch.Tensor | None = None,
    order=5,
):
    mu = feat_flat
    var = torch.full_like(mu, noise_var)

    use_multi = bool(
        getattr(model_ref, "use_multi_head", getattr(model_ref, "use_multi_qh", False))
    )
    if not use_multi:
        for block in model_ref.upscale:
            mu, var = forward_adf_scaleblock(block, mu, var, order)
        return mu, var

    if forg_type_ids is None:
        raise ValueError("forg_type_ids is required for multi-head adf_upscale")

    forg_type_ids = forg_type_ids.to(device=feat_flat.device, dtype=torch.long).view(-1)
    if forg_type_ids.numel() != feat_flat.shape[0]:
        raise ValueError(
            f"forg_type_ids must have shape ({feat_flat.shape[0]},), got {tuple(forg_type_ids.shape)}"
        )

    mu_out = None
    var_out = None
    for expert_id in forg_type_ids.unique(sorted=True).tolist():
        idx = (forg_type_ids == expert_id).nonzero(as_tuple=True)[0]
        mu_e = mu[idx]
        var_e = var[idx]
        for block in model_ref.upscale_heads[expert_id]:
            mu_e, var_e = forward_adf_scaleblock(block, mu_e, var_e, order)
        if mu_out is None:
            mu_out = torch.empty(
                feat_flat.shape[0],
                *mu_e.shape[1:],
                device=mu_e.device,
                dtype=mu_e.dtype,
            )
            var_out = torch.empty(
                feat_flat.shape[0],
                *var_e.shape[1:],
                device=var_e.device,
                dtype=var_e.dtype,
            )
        mu_out[idx] = mu_e
        var_out[idx] = var_e

    return mu_out, var_out
