"""Localize one pre-cropped 512x512 RGB image without a ground-truth mask."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from rise.config import ModelConfig
from rise.checkpoint import load_model
from rise.data import read_jpeg
from rise.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Required for original research checkpoints")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/prediction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode", choices=["sequential", "parallel"], default="sequential"
    )
    parser.add_argument("--gh-order", type=int, choices=[3, 5, 7], default=5)
    args = parser.parse_args()
    config = ModelConfig.read(args.config) if args.config else None
    model, metadata = load_model(args.checkpoint, config, args.device)
    sample = read_jpeg(Path(args.image).read_bytes(), model.use_dct_feature)
    tensors = {key: value.unsqueeze(0).to(args.device) for key, value in sample.items()}
    result = predict(
        model, tensors.pop("image"), **tensors, gh_order=args.gh_order, mode=args.mode
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    probability = result["probability"][0].cpu().numpy()
    np.save(output / "probability.npy", probability)
    Image.fromarray(((probability >= 0.5) * 255).astype(np.uint8)).save(
        output / "mask.png"
    )
    metadata.update(
        image=args.image,
        gh_order=args.gh_order,
        noise_variance=0.5,
        mode=args.mode,
        expert_id=int(result["expert_id"][0]),
        confidence=result["confidence"][0].tolist(),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved mask and probability to {output}")


if __name__ == "__main__":
    main()
