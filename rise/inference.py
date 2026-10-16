"""Ground-truth-free Latent Perturbation Confidence (LPC) expert selection."""

import torch
from torch.nn import functional as F
from .models.adf import adf_upscale


def query_probability(mask_logits, class_logits, size):
    masks = F.interpolate(
        mask_logits, size=size, mode="bilinear", align_corners=False
    ).sigmoid()
    query = class_logits.softmax(-1)[..., 0].argmax(dim=1)
    return masks[torch.arange(masks.shape[0], device=masks.device), query]


def soft_dice_confidence(probability):
    p = probability.flatten(1)
    y = (p >= 0.5).to(p.dtype)
    return 2 * (p * y).sum(1) / (p.sum(1) + y.sum(1) + 1e-8)


@torch.no_grad()
def predict(
    model, image, dct=None, qt=None, gh_order=5, noise_variance=0.5, mode="sequential"
):
    """Return selected RGB prediction, expert ids, confidence and per-expert maps.

    ADF acts only on dense features after the VFM. The final mask is the selected
    expert's original unperturbed mask, not the ADF-averaged prediction.
    Sequential mode reduces GPU memory; parallel mode batches all experts.
    """
    if gh_order not in (3, 5, 7) or noise_variance < 0:
        raise ValueError("Unsupported quadrature order or negative noise variance")
    if mode not in ("sequential", "parallel"):
        raise ValueError("mode must be sequential or parallel")
    batch, experts = image.shape[0], model.num_forg_type
    maps, confidences = [], []
    amp = image.device.type == "cuda"
    if mode == "parallel":
        ids = torch.arange(experts, device=image.device).repeat(batch)
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp):
            masks, classes, queries, features = model(
                image.repeat_interleave(experts, 0),
                ids,
                dct=dct.repeat_interleave(experts, 0) if dct is not None else None,
                qt=qt.repeat_interleave(experts, 0) if qt is not None else None,
                return_features=True,
            )
        original = query_probability(
            masks[-1].float(), classes[-1].float(), image.shape[-2:]
        ).reshape(batch, experts, *image.shape[-2:])
        queries = queries.reshape(batch, experts, *queries.shape[1:])
        features = features.reshape(batch, experts, *features.shape[1:])
    for expert in range(experts):
        ids = torch.full((batch,), expert, dtype=torch.long, device=image.device)
        if mode == "sequential":
            with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp):
                masks, classes, query, feature = model(
                    image, ids, dct=dct, qt=qt, return_features=True
                )
            probability = query_probability(
                masks[-1].float(), classes[-1].float(), image.shape[-2:]
            )
        else:
            probability, query, feature = (
                original[:, expert],
                queries[:, expert],
                features[:, expert],
            )
        maps.append(probability)
        if experts == 1:
            confidences.append(soft_dice_confidence(probability))
            continue
        # FP32 moments; process one expert at a time to bound ADF memory.
        mu, var = adf_upscale(
            model, feature.float(), noise_variance, ids, order=gh_order
        )
        q = model.apply_mask_head(query.float(), ids).float()
        logits_mu = torch.einsum("bqc,bchw->bqhw", q, mu)
        logits_var = torch.einsum("bqc,bchw->bqhw", q.square(), var).clamp_min(0)
        std = (logits_var + 1e-8).sqrt()
        expected = 0.5 * ((logits_mu - std).sigmoid() + (logits_mu + std).sigmoid())
        confidences.append(soft_dice_confidence(expected.mean(1)))
    maps = torch.stack(maps, 1)
    confidence = torch.stack(confidences, 1)
    selected = confidence.argmax(1)
    return {
        "probability": maps[torch.arange(batch, device=image.device), selected],
        "expert_id": selected,
        "confidence": confidence,
        "expert_probability": maps,
    }


def binary_f1(probability, target):
    pred, target = (probability >= 0.5).flatten(1), target.bool().flatten(1)
    intersection = (pred & target).sum(1).float()
    precision = intersection / (pred.sum(1) + 1e-8)
    recall = intersection / (target.sum(1) + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)
