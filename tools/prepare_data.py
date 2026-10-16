"""Create portable manifests without copying images or distributing datasets."""

import argparse
import json
import pickle
from pathlib import Path

TYPE_ID = {"GE": 0, "CM": 1, "SP": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lm = sub.add_parser(
        "doctamper", help="LMDB split -> JSONL; official QFs or training labels"
    )
    lm.add_argument("--root", required=True)
    lm.add_argument(
        "--lmdb-dir", default=".", help="LMDB parent directory relative to root"
    )
    lm.add_argument(
        "--split", choices=["TrainingSet", "TestingSet", "FCD", "SCD"], required=True
    )
    lm.add_argument(
        "--jpeg-records", help="Official DocTamperV1-{split}_75.pk for evaluation"
    )
    lm.add_argument("--num-experts", type=int, choices=[1, 3, 9], default=1)
    lm.add_argument(
        "--forgery-types",
        help="Pickle list/dict mapping zero-based indices to GE/CM/SP",
    )
    lm.add_argument(
        "--assignments",
        help="JSON mapping zero-based LMDB indices to expert ids (RiSE-9)",
    )
    lm.add_argument("--output", required=True)
    cross = sub.add_parser("cross", help="Original path-list pickle -> portable JSONL")
    cross.add_argument("--root", required=True)
    cross.add_argument(
        "--paths",
        required=True,
        help="Pickle of (image_path, mask_path[, ...]) records",
    )
    cross.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    records = []
    if args.command == "cross":
        with open(args.paths, "rb") as handle:
            paths = pickle.load(handle)
        for image, mask, *_ in paths:

            def relative(path):
                if path is None:
                    return None
                path = Path(path)
                return str(
                    (path if path.is_absolute() else root / path)
                    .resolve()
                    .relative_to(root)
                )

            records.append({"image": relative(image), "mask": relative(mask)})
    else:
        import lmdb

        path = root / args.lmdb_dir / f"DocTamperV1-{args.split}"
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            count = int(txn.get(b"num-samples"))
        env.close()
        qfs = types = ids = None
        if args.jpeg_records:
            with open(args.jpeg_records, "rb") as handle:
                qfs = pickle.load(handle)
        if args.forgery_types:
            with open(args.forgery_types, "rb") as handle:
                types = pickle.load(handle)
        if args.assignments:
            ids = json.loads(Path(args.assignments).read_text())
        training = args.split == "TrainingSet"
        if not training and qfs is None:
            parser.error(
                "Evaluation splits require --jpeg-records for record[-1] compression"
            )
        if training and args.num_experts == 3 and types is None:
            parser.error(
                "RiSE-3 requires --forgery-types; it uses type metadata, not global clusters"
            )
        if training and args.num_experts == 9 and ids is None:
            parser.error(
                "RiSE-9 requires --assignments (three balanced clusters within each forgery type)"
            )
        for index in range(count):
            row = {
                "lmdb": str(path.resolve().relative_to(root)),
                "index": index,
                "name": f"{index:06d}",
            }
            if qfs is not None:
                row["jpeg_record"] = [int(q) for q in qfs[index]]
            if training:
                row["expert_id"] = (
                    0
                    if args.num_experts == 1
                    else TYPE_ID[types[index]]
                    if args.num_experts == 3
                    else int(ids[str(index)])
                )
                if not 0 <= row["expert_id"] < args.num_experts:
                    raise ValueError(f"Invalid expert id at image {index}")
            records.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in records))
    print(f"Wrote {len(records)} samples to {output}")


if __name__ == "__main__":
    main()
