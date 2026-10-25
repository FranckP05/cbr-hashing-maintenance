# exp_falconn — Cross-Polytope LSH Experiment

Second run of the LSH study. Evaluates the **CrossPolytope LSH**
(FALCONN-style, implemented from scratch in pure NumPy) against the
**linear exact** baseline on the **normalised cosine arm**.

## Research Question

> Can a LSH correctly adapted to the metric used reduce search time while preserving the quality of returned neighbours and CBR classification quality?

This run tests the strongest theoretical candidate from the phase-1 SOTA review:
the **cross-polytope LSH**, which is *optimal* for angular (cosine) distance
[Andoni et al., 2015] and is the core of the FALCONN library.

## Datasets

| Dataset        | Total size | Dimension | Role                        |
|----------------|-----------|-----------|------------------------------|
| Heart Disease  | 297        | 13        | Small control dataset        |
| ChestX-ray14   | 112 120    | 1024      | Main large-scale dataset     |

Both are **L2-normalised** (cosine arm, protocol §5): on the unit sphere
`‖p−q‖² = 2 − 2·cos(p,q)`, so exact L2 ranking is identical to cosine ranking.

## Methods Compared (cosine arm)

1. **Linear exact** — exhaustive L2 baseline on the unit sphere (BLAS-optimised)
2. **CrossPolytope** — `2·idx + sign_bit` hashing after `H·Dₙ·H·Dₙ₋₁·…·H·D₁`
   rotations, vectorised fast Hadamard transform, multi-probe, `last_cp_dim` remainder

Phase-1 methods (SimHash, E2LSH from `exp_lsh/`) are **not re-run**; their
results are reused for the final comparison.

## Quick Start

```bash
pip install -r requirements.txt
# Data checks (cosine-arm sanity)
jupyter notebook notebooks/00_data_checks.ipynb
# Calibration (validation only)
jupyter notebook notebooks/01_calibration.ipynb
# Benchmark (test set)
jupyter notebook notebooks/02_benchmark.ipynb
```

## Project Layout

```
exp_falconn/
├── README.md
├── build_notebooks.py  # generator for the reference notebooks
├── config/             # Protocol constants (seeds, thresholds, cosine metric)
├── src/
│   ├── datasets/       # Loaders + L2-normalisation (cosine arm)
│   ├── indexes/        # linear_exact, cross_polytope
│   ├── calibration/    # calibrate_cross_polytope grid search (val only)
│   ├── evaluation/     # Recall@k, nDCG, CBR metrics
│   └── timing/         # Benchmarking harness
├── tests/              # 21 tests (FHT, rotation, hash, multi-probe, recall)
├── notebooks/          # 00_data_checks … 04_analysis
├── results/            # Output CSVs, figures, run registry
└── docs/               # Progress tracking and task board
```

## Protocol Reference

See `../draft/protocol.md` for the full experimental protocol.

## Scientific References

- Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., Schmidt, L. — *Practical and Optimal LSH for Angular Distance*, NeurIPS 2015.
- Andoni, A., Razenshteyn, I., Avdiukhin, D. — *FALCONN: Similarity Search over Massive Data*, 2017.
- Lv, Q. et al. — *Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search*, 2007.
- Charikar, M. S. — *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002.
