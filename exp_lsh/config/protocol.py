"""
Protocol constants for exp_lsh.

Defines the quality thresholds and success criteria described in sections 8 and 12
of the experimental protocol.
"""

# --------------------------------------------------------------------------- #
# Calibration quality constraints (section 8)
# --------------------------------------------------------------------------- #
MIN_RECALL_AT_10: float = 0.95       # Recall@10 >= 0.95 on validation set
MAX_F1_DROP: float = 0.01            # macro-F1 drop vs. linear <= 0.01

# --------------------------------------------------------------------------- #
# Success criteria (section 12)
# --------------------------------------------------------------------------- #
SUCCESS_RECALL_AT_10: float = 0.95
SUCCESS_MAX_F1_DROP: float = 0.01    # macro-F1 >= linear - 0.01

# --------------------------------------------------------------------------- #
# Timing protocol (section 9.3)
# --------------------------------------------------------------------------- #
TIMING_WARMUP: int = 1               # number of warm-up query repetitions
TIMING_REPEATS: int = 5             # number of measured repetitions
# Aggregation: median over TIMING_REPEATS is kept.

# --------------------------------------------------------------------------- #
# Splits (section 6)
# --------------------------------------------------------------------------- #
TEST_FRACTION: float = 0.20          # 20 % held-out test
VALID_FRACTION_OF_TRAIN: float = 0.20  # 20 % of remaining 80 % → validation
# Final approximate proportions: 64 % train / 16 % val / 20 % test

# --------------------------------------------------------------------------- #
# Number of neighbours
# --------------------------------------------------------------------------- #
K_NEIGHBOURS: int = 10
