"""Serializable model settings; no import-time global experiment state."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    num_experts: int = 9
    residual_rank: int = 1
    num_queries: int = 1
    query_mode: str = "multi"
    use_dct: bool = False
    dct_fuse_block_idx: int = 10  # Zero-based: fuse before transformer block 11.
    dct_fuse_prob: float = 0.5  # Training only; inference always uses 1.0.

    def __post_init__(self):
        if self.num_experts not in (1, 3, 9) or self.num_queries != 1:
            raise ValueError(
                "Main-paper variants use 1/3/9 experts and one query per expert"
            )
        if self.query_mode not in ("single", "multi"):
            raise ValueError("query_mode must be single or multi")
        if self.use_dct and self.dct_fuse_block_idx != 10:
            raise ValueError("The released DCT architecture fuses at block index 10")
        if not 0 <= self.dct_fuse_prob <= 1:
            raise ValueError("dct_fuse_prob must be between 0 and 1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))
