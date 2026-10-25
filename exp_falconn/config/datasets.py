"""
Dataset configuration for exp_lsh.

Two datasets are used:
  - Heart Disease  (n=297, d=13)  — small control dataset
  - ChestX-ray14  (n=112120, d=1024) — main large-scale dataset

Diabetes-130 is explicitly excluded (class imbalance / HeCBR presence).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    name: str
    dimension: int
    total_size: int
    # Sizes of the retrieval base to benchmark.
    # None means "full train set".
    base_sizes: List[Optional[int]]
    # Maximum number of test queries to use (None = all).
    max_queries: Optional[int]


HEART_DISEASE = DatasetConfig(
    name="heart_disease",
    dimension=13,
    total_size=297,
    base_sizes=[None],          # full train only
    max_queries=None,           # all test samples
)

CHEST_XRAY14 = DatasetConfig(
    name="chest_xray14",
    dimension=1024,
    total_size=112_120,
    base_sizes=[10_000, 50_000, None],  # 10k, 50k, full train
    max_queries=300,            # 300 reproducible test queries per split
)

ALL_DATASETS = [HEART_DISEASE, CHEST_XRAY14]
