"""Strict loading for original DDP checkpoints and new portable checkpoints."""

from pathlib import Path
import torch
from .config import ModelConfig
from .models import RiSE


def load_model(checkpoint_path, config=None, device="cuda"):
    # Load only checkpoints from trusted sources: historical .pth files use pickle.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("model", checkpoint)
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    if config is None:
        if "model_config" not in checkpoint:
            raise ValueError(
                "Original checkpoints require an explicit --config; fusion index is not stored in weights"
            )
        config = ModelConfig(**checkpoint["model_config"])
    expert_shapes = {
        tuple(v.shape) for k, v in weights.items() if k.endswith(".expert_S")
    }
    if expert_shapes != {(config.num_experts, config.residual_rank)}:
        raise ValueError(
            f"Expert shape mismatch: {expert_shapes}; check the model config"
        )
    query_rows = config.num_queries * (
        config.num_experts if config.query_mode == "multi" else 1
    )
    if weights["q_table.weight"].shape[0] != query_rows:
        raise ValueError("Checkpoint query sharing does not match the config")
    has_dct = "dct_encoder.projector.0.weight" in weights
    if has_dct != config.use_dct:
        raise ValueError("Checkpoint DCT branch does not match the config")
    if has_dct:
        channels = weights["dct_encoder.projector.0.weight"].shape[1]
        expected = 1024
        if channels != expected:
            raise ValueError(
                f"DCT projector expects {channels} channels; config selects {expected}"
            )
    model = RiSE(config, initialize=False)
    model.load_state_dict(
        weights, strict=True
    )  # Never skip mismatched or missing weights.
    model.dct_fuse_prob = 1.0
    model = model.to(device).eval()
    return model, {
        "checkpoint": str(Path(checkpoint_path)),
        "loaded_tensors": len(weights),
        "load_mode": "strict",
        "model_config": config.to_dict(),
    }
