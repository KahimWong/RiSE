"""Portable JSONL datasets for DocTamper LMDBs and image/mask folders.

Evaluation never silently resizes or recompresses cross-domain JPEGs. In-domain
samples are JPEG-encoded once at record[-1], not through the whole QF history.
"""

import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_jpeg(raw, use_dct=False):
    with Image.open(io.BytesIO(raw)) as source:
        if source.size != (512, 512):
            raise ValueError(f"Expected pre-cropped 512x512 input, got {source.size}")
        image = (
            torch.from_numpy(np.asarray(source.convert("RGB")).copy())
            .permute(2, 0, 1)
            .float()
        )
        if not use_dct:
            return {"image": image}
        if source.format != "JPEG":
            raise ValueError(
                "DCT inference requires an original JPEG, not a synthetic re-encode"
            )
        qt = np.asarray(source.quantization[0], dtype=np.int64).reshape(8, 8)
        if qt.min() < 0 or qt.max() >= 64:
            raise ValueError(
                "JPEG quantization values exceed the trained DCT vocabulary [0, 63]"
            )
    from jpeg2dct.numpy import loads

    coefficients, _, _ = loads(raw, normalized=False)
    rows, cols, _ = coefficients.shape
    dct = (
        coefficients.reshape(rows, cols, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(rows * 8, cols * 8)
    )
    return {
        "image": image,
        "dct": torch.from_numpy(np.clip(np.abs(dct.astype(np.int32)), 0, 20)).long(),
        "qt": torch.from_numpy(qt.copy()).long(),
    }


def encode_jpeg(raw, quality):
    with Image.open(io.BytesIO(raw)) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
        return buffer.getvalue()


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest,
        root=".",
        protocol="original",
        use_dct=False,
        limit=None,
        num_experts=1,
        training=False,
        min_quality=90,
    ):
        self.records = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError("Manifest contains no records")
        self.root, self.protocol, self.use_dct = Path(root), protocol, use_dct
        self.num_experts, self.training = num_experts, training
        self.min_quality = torch.tensor(min_quality, dtype=torch.int64).share_memory_()
        self._environments = {}  # LMDB handles are opened lazily in each worker process.
        if protocol not in ("original", "final_record", "train"):
            raise ValueError(f"Unknown JPEG protocol: {protocol}")
        for row in self.records:
            if training and num_experts > 1 and "expert_id" not in row:
                raise ValueError(
                    "Every multi-expert training sample needs an offline expert_id"
                )
            if not 0 <= int(row.get("expert_id", 0)) < num_experts:
                raise ValueError("Manifest expert_id is outside the configured range")
            if protocol == "final_record" and not row.get("jpeg_record"):
                raise ValueError(
                    "Every in-domain evaluation sample needs its official jpeg_record"
                )

    def __len__(self):
        return len(self.records)

    def _read(self, row):
        if "lmdb" not in row:
            image = (self.root / row["image"]).read_bytes()
            mask = (self.root / row["mask"]).read_bytes() if row.get("mask") else None
            return image, mask
        import lmdb

        path = str(self.root / row["lmdb"])
        if path not in self._environments:
            self._environments[path] = lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False
            )
        with self._environments[path].begin() as txn:
            image = txn.get(f'image-{int(row["index"]):09d}'.encode())
            mask = txn.get(f'label-{int(row["index"]):09d}'.encode())
        if image is None or mask is None:
            raise ValueError(f'Missing LMDB image/label at index {row["index"]}')
        return image, mask

    def __getitem__(self, index):
        row = self.records[index]
        raw, mask_raw = self._read(row)
        qf = -1
        if self.protocol == "final_record":
            qf = int(row["jpeg_record"][-1])
            raw = encode_jpeg(raw, qf)
        elif self.protocol == "train":
            # Fresh augmentation: 1--3 sequential JPEGs, as in the research loader.
            for _ in range(int(torch.randint(1, 4, ()).item())):
                qf = int(torch.randint(int(self.min_quality.item()), 101, ()).item())
                raw = encode_jpeg(raw, qf)
        item = read_jpeg(raw, self.use_dct)
        if mask_raw is None:
            mask = np.zeros((512, 512), dtype=np.uint8)
        else:
            with Image.open(io.BytesIO(mask_raw)) as source:
                mask = (np.asarray(source.convert("L")) != 0).astype(np.uint8)
            if mask.shape != (512, 512):
                raise ValueError(
                    "Mask must match the 512x512 image; no implicit resizing"
                )
        item.update(
            mask=torch.from_numpy(mask.copy()).long(),
            expert_id=int(row.get("expert_id", 0)),
            index=index,
            qf=qf,
            name=str(row.get("name", row.get("image", row.get("index", index)))),
        )
        return item
