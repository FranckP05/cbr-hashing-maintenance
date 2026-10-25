"""
Metric definitions and evaluation settings (sections 3, 5 and 10).
"""

# Recall@k values to report.
RECALL_KS = [1, 5, 10]

# Primary metric for exp_falconn: the cosine arm (section 5).
# Cross-polytope LSH targets angular / cosine similarity, so all vectors
# are L2-normalised and ranked by squared L2 distance, which on the unit
# sphere is equivalent to cosine ranking:  ||p - q||^2 = 2 - 2*cos(p, q).
PRIMARY_METRIC = "cosine"         # "l2" | "cosine"

# The Chest-cosine arm is now the object of this experiment: the input
# vectors are L2-normalised and the baseline is an exact cosine search.
CHEST_COSINE_ARM_ENABLED: bool = True

# CBR vote rule used uniformly for all methods.
CBR_VOTE_RULE = "majority"        # majority vote over k nearest neighbours
