"""Extract RiSE-1 output-query embeddings for offline expert partitioning."""

import argparse
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from rise.checkpoint import load_model
from rise.config import ModelConfig
from rise.data import ManifestDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/rise1.json")
    parser.add_argument(
        "--manifest", required=True, help="Full training manifest, with expert_id=0"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--forgery-types", required=True, help="Official GE/CM/SP mapping by LMDB index"
    )
    parser.add_argument("--output", default="outputs/training_queries.npz")
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint, ModelConfig.read(args.config))
    if model.num_forg_type != 1 or model.use_dct_feature:
        raise ValueError("Partition features must come from the RGB RiSE-1 model")
    dataset = ManifestDataset(args.manifest, args.data_root, protocol="original")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    with open(args.forgery_types, "rb") as handle:
        types = pickle.load(handle)
    embeddings, indices, labels = [], [], []
    with torch.inference_mode():
        for sample in loader:
            image = sample["image"].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                _, _, query = model(
                    image, torch.zeros(1, dtype=torch.long, device="cuda")
                )
            position = int(sample["index"][0])
            record = dataset.records[position]
            index = int(record["index"])
            embeddings.append(query[0, 0].float().cpu().numpy())
            indices.append(index)
            labels.append(types[index])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=np.stack(embeddings),
        indices=np.asarray(indices),
        types=np.asarray(labels),
    )
    print(f"Saved {len(indices)} query embeddings to {output}")


if __name__ == "__main__":
    main()
