"""
Metric definitions and evaluation settings (sections 3 and 10).
"""

# Recall@k values to report.
RECALL_KS = [1, 5, 10]

# Primary metric: squared L2 distance preserves neighbour ranking.
PRIMARY_METRIC = "l2"          # "l2" | "cosine"

# Optional Chest-cosine arm: activate only when cosinus similarity is
# scientifically justified for ChestX-ray14 embeddings.
CHEST_COSINE_ARM_ENABLED: bool = False

# CBR vote rule used uniformly for all methods.
CBR_VOTE_RULE = "majority"     # majority vote over k nearest neighbours
