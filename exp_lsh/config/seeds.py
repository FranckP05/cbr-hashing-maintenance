"""
Seeds used across the experiment (section 6 of the protocol).

Two categories of seeds are distinguished:

  SPLIT_SEEDS  — control train/val/test stratified splits.
  HASH_SEEDS   — control random projections inside LSH indexes.

SEED_SAMPLING (1337) is reserved for reproducible sub-base sampling
and must not be used as a sixth experimental seed.
"""

from typing import List

# Five split seeds.
SPLIT_SEEDS: List[int] = [42, 7, 123, 2024, 99]

# Three LSH hash seeds evaluated per split.
HASH_SEEDS: List[int] = [11, 29, 43]

# Reserved for sub-base sampling only — not an experimental seed.
SEED_SAMPLING: int = 1337
