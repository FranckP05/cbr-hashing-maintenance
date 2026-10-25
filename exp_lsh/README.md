# exp_lsh — LSH Experiment for Approximate CBR Search

This experiment evaluates **Locality-Sensitive Hashing (LSH)** as an approximate nearest-neighbour retrieval method within a medical CBR system.

## Research Question

> Can a LSH correctly adapted to the metric used reduce search time while preserving the quality of returned neighbours and CBR classification quality?

## Datasets

| Dataset        | Total size | Dimension | Role                        |
|----------------|-----------|-----------|------------------------------|
| Heart Disease  | 297        | 13        | Small control dataset        |
| ChestX-ray14   | 112 120    | 1024      | Main large-scale dataset     |

## Methods Compared (L2 arm)

1. **Linear exact** — exhaustive L2 baseline (BLAS-optimised)
2. **SimHash** — random hyperplane, kept as diagnostic ablation (angular metric)
3. **E2LSH p-stable** — main proposed method for L2 distance

## Quick Start

```bash
pip install -r requirements.txt
# Data checks
jupyter notebook notebooks/00_data_checks.ipynb
# Calibration
jupyter notebook notebooks/01_calibration.ipynb
# Benchmark
jupyter notebook notebooks/02_benchmark.ipynb
```

## Project Layout

```
exp_lsh/
├── README.md
├── config/          # Protocol constants (datasets, seeds, metrics, thresholds)
├── src/
│   ├── datasets/    # Loaders and preprocessors
│   ├── indexes/     # linear_exact, simhash, e2lsh
│   ├── calibration/ # Grid search on validation set
│   ├── evaluation/  # Recall@k, nDCG, CBR metrics
│   └── timing/      # Benchmarking harness
├── tests/           # Unit & integration tests
├── notebooks/       # Exploratory and reporting notebooks
├── results/         # Output CSVs, figures, run registry
└── docs/            # Progress tracking and task board
```

## Protocol Reference

See `../draft/protocol.md` for the full experimental protocol.

## Scientific References

- Charikar, M. S. — *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002.
- Datar, M., Indyk, P., Immorlica, N., Mirrokni, V. — *LSH Scheme Based on p-Stable Distributions*, 2004.
- Lv, Q. et al. — *Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search*, 2007.
- Andoni, A. et al. — *Practical and Optimal LSH for Angular Distance*, 2015.
