"""Evaluate all seven Doc Protocol splits; report per-image mean F1 and routing."""

import argparse
import csv
import json
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from rise.config import ModelConfig
from rise.checkpoint import load_model
from rise.data import ManifestDataset
from rise.inference import binary_f1, predict

IN_DOMAIN = ("FCD", "SCD", "TestingSet")
CROSS_DOMAIN = ("T-SROIE", "OSTF", "TPIC-13", "RTM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "--data", required=True, help="JSON with root and datasets mapping"
    )
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        help="First N images per split; omitted means the complete split",
    )
    parser.add_argument("--gh-order", type=int, choices=[3, 5, 7], default=5)
    parser.add_argument(
        "--mode", choices=["sequential", "parallel"], default="sequential"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume CSVs from the same checkpoint/config/protocol",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    torch.set_num_threads(8)
    model, metadata = load_model(
        args.checkpoint,
        ModelConfig.read(args.config) if args.config else None,
        args.device,
    )
    torch.set_num_threads(1)
    data_file = Path(args.data).resolve()
    data = json.loads(data_file.read_text())
    root = data_file.parent / data["root"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata.update(
        gh_order=args.gh_order,
        noise_variance=0.5,
        mode=args.mode,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        data_config=data,
        in_domain_protocol="one JPEG encode at record[-1]",
        cross_domain_protocol="original RGB JPEG; no recompression",
    )
    metadata_path = output / "metadata.json"
    if (
        args.resume
        and metadata_path.exists()
        and json.loads(metadata_path.read_text()) != metadata
    ):
        raise ValueError("Resume metadata differs; choose a new output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    report = {}
    for name in IN_DOMAIN + CROSS_DOMAIN:
        if name not in data["datasets"]:
            raise ValueError(f"Missing dataset {name}; supply all seven splits")
        manifest = data_file.parent / data["datasets"][name]
        dataset = ManifestDataset(
            manifest,
            root,
            "final_record" if name in IN_DOMAIN else "original",
            model.use_dct_feature,
            limit=args.limit,
        )
        destination = output / f"{name}.csv"
        rows = []
        if args.resume and destination.exists():
            with destination.open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["index"] = int(row["index"])
                row["f1"], row["oracle_f1"] = float(row["f1"]), float(row["oracle_f1"])
        done = {row["index"] for row in rows}
        if len(done) != len(rows):
            raise ValueError(f"Duplicate resumed indices in {destination}")
        expected = set(range(args.shard_index, len(dataset), args.num_shards))
        if not done.issubset(expected):
            raise ValueError("Resumed indices do not belong to this shard")
        indices = sorted(expected - done)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )
        started = time.monotonic()
        with destination.open(
            "a" if args.resume else "w", newline="", buffering=1
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "name",
                    "qf",
                    "f1",
                    "oracle_f1",
                    "expert_id",
                    "oracle_expert_id",
                ],
            )
            if destination.stat().st_size == 0:
                writer.writeheader()
            for offset, item in enumerate(loader):
                image = item["image"].to(args.device)
                inputs = {
                    k: item[k].to(args.device) for k in ("dct", "qt") if k in item
                }
                result = predict(
                    model, image, **inputs, gh_order=args.gh_order, mode=args.mode
                )
                mask = item["mask"].to(args.device)
                score = binary_f1(result["probability"], mask)
                experts = result["expert_probability"]
                expert_f1 = binary_f1(
                    experts.flatten(0, 1), mask.repeat_interleave(experts.shape[1], 0)
                )
                row = dict(
                    index=int(item["index"][0]),
                    name=item["name"][0],
                    qf=int(item["qf"][0]),
                    f1=float(score[0]),
                    oracle_f1=float(expert_f1.max()),
                    expert_id=int(result["expert_id"][0]),
                    oracle_expert_id=int(expert_f1.argmax()),
                )
                rows.append(row)
                writer.writerow(row)
                if offset % 50 == 0:
                    print(
                        f"{name} shard {args.shard_index}: {len(rows)}/{len(expected)}, {(time.monotonic()-started)/(offset+1):.3f}s/image",
                        flush=True,
                    )
        report[name] = {
            "samples": len(rows),
            "expected": len(expected),
            "f1": sum(r["f1"] for r in rows) / len(rows) if rows else None,
            "oracle_f1": sum(r["oracle_f1"] for r in rows) / len(rows)
            if rows
            else None,
        }
        print(name, report[name], flush=True)
    report["scope"] = (
        "full"
        if args.num_shards == 1
        else "shard means; aggregate per-image CSVs for full-split means"
    )
    report["AvgI"] = sum(report[n]["f1"] for n in IN_DOMAIN) / 3
    report["AvgC"] = sum(report[n]["f1"] for n in CROSS_DOMAIN) / 4
    report["Avg"] = sum(report[n]["f1"] for n in IN_DOMAIN + CROSS_DOMAIN) / 7
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("AvgI", "AvgC", "Avg")}, indent=2))


if __name__ == "__main__":
    main()
