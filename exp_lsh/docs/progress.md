# Progress Log — exp_lsh

> **Purpose:** Chronological record of all work done on this experiment.
> Any agent picking up this project should read this file first to understand the current state.

---

## 2026-08-16 — Initial scaffold

**Agent:** Antigravity (conversation `97f21a6c-69e0-44e7-9e47-474511cf7f8d`)
**Protocol source:** `../draft/protocol.md`

### What was done

The full project scaffold was generated from the experimental protocol (`protocol.md`, sections 1–17).

#### Directory structure created

```
exp_lsh/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── datasets.py       # DatasetConfig for Heart Disease & ChestX-ray14
│   ├── protocol.py       # Quality thresholds, timing constants, split fractions
│   ├── seeds.py          # SPLIT_SEEDS, HASH_SEEDS, SEED_SAMPLING
│   └── metrics.py        # RECALL_KS, PRIMARY_METRIC, CBR_VOTE_RULE
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── heart_disease.py   # Loader + StandardScaler pipeline
│   │   └── chest_xray14.py    # Embedding loader + sub-base sampling
│   ├── indexes/
│   │   ├── __init__.py
│   │   ├── linear_exact.py    # BLAS-optimised exhaustive L2 search
│   │   ├── simhash.py         # Random-hyperplane (diagnostic ablation)
│   │   └── e2lsh.py           # E2LSH p-stable with multi-probe
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── grid_search.py     # Validation-set calibration + best-config selection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Recall@k, nDCG, macro-F1, AUC, CBR predictions
│   └── timing/
│       ├── __init__.py
│       ├── harness.py         # timed_query() with warm-up + median aggregation
│       └── system_info.py     # Python/NumPy/sklearn/FAISS/CPU/RAM info
├── tests/
│   ├── __init__.py
│   ├── test_splits.py         # Split overlap, NaN/Inf checks
│   ├── test_linear.py         # LinearExact correctness
│   └── test_e2lsh.py          # E2LSH correctness + recall smoke test
├── notebooks/
│   ├── 00_data_checks.ipynb
│   ├── 01_calibration.ipynb
│   ├── 02_benchmark.ipynb
│   ├── 03_scaling.ipynb
│   └── 04_analysis.ipynb
├── results/                   # (empty — populated at run time)
└── docs/
    ├── progress.md            # ← this file
    └── tasks.md               # Task board
```

#### Key design decisions

| Decision | Rationale |
|---|---|
| Squared L2 used for ranking | Preserves neighbour order; avoids sqrt (protocol §3) |
| `SEED_SAMPLING = 1337` reserved | Must not be used as a 6th experimental seed (protocol §6) |
| SimHash marked as *ablation* | It targets cosine, not L2 — kept for diagnostics only (§4.2) |
| No fallback exact search | Silent fallback is forbidden in the main arm (§10) |
| Calibration on val only | Test data must never influence any parameter choice (§8) |
| Tiebreaker order: latency → candidate fraction → memory → build time | Directly from protocol §8 |

---

## 2026-08-16 — Notebooks fully rewritten

**Agent:** Antigravity (conversation `97f21a6c-69e0-44e7-9e47-474511cf7f8d`)

All 5 notebooks replaced with production-quality implementations. Previous stubs had only 2–3 placeholder lines.

| Notebook | Cells | Key additions |
|---|---|---|
| `00_data_checks.ipynb` | 11 | Real split checks, NaN/Inf, overlap assertions, profile CSV export |
| `01_calibration.ipynb` | 11 | Vectorized E2LSH build, tqdm grid search, val-only calibration, JSON param export |
| `02_benchmark.ipynb` | 13 | Ground-truth caching (.npz), 1-warmup/5-rep timing, 4 output CSVs |
| `03_scaling.ipynb` | 11 | Latency/Recall/Candidates/BuildTime scaling plots, speedup table |
| `04_analysis.ipynb` | 18 | Wilcoxon + bootstrap CI, 8 protocol figures, outcome classification, summary.json |

**Kaggle parallel strategy decided:** notebooks 01+02 are parametric on `SPLIT_SEED` — duplicate 5×, one per seed, run simultaneously. Phase 0 (data checks) runs once. Phase 2 (analysis) waits for all Phase 1 sessions.

**Vectorized E2LSH build** added inline in notebooks 01 and 02: hashes all base vectors at once via `X_base @ A.T`, avoiding the Python loop over 90k items that would have been prohibitively slow on ChestX-ray14.

---

## 2026-08-16 — Phase 0 Data Validation Completed

**Agent:** Antigravity (conversation `97f21a6c-69e0-44e7-9e47-474511cf7f8d`)

1. **Unit Tests:** `pytest tests/` executed with 16/16 tests passing (E2LSH determinism, seed variance, non-negativity, LinearExact top-1 self-retrieval, split non-overlap).
2. **Data Validation (Phase 0):** Full check across all 5 seeds (`[42, 7, 123, 2024, 99]`):
   - **Heart Disease ($n=297, d=13$):** Train: 189, Val: 48, Test: 60 (exact 64/16/20 split). StandardScaler fit exclusively on train. Zero NaN/Inf, zero overlap across all 5 splits.
   - **ChestX-ray14 ($n=112120, d=1024$):** Train: 71,756, Val: 17,940, Test: 300 queries per split. Sub-base sampling verified (10k, 50k, full). Zero NaN/Inf, zero overlap.
   - Generated files: `results/data_checks/dataset_profile.csv`, `results/data_checks/split_profile.csv`.

---

## 2026-08-16 — Phase 1: Kaggle Parallel Dispatch

**Agent:** Antigravity (conversation `97f21a6c-69e0-44e7-9e47-474511cf7f8d`)

1. **Code Package:** Created and uploaded [`franckyoteu/exp-lsh-code`](https://www.kaggle.com/datasets/franckyoteu/exp-lsh-code) to Kaggle.
2. **Parallel Kernels Dispatched:** 5 independent kernels pushed and running in parallel on Kaggle:
   - [Seed 42](https://www.kaggle.com/code/franckyoteu/exp-lsh-pipeline-seed-42) — `RUNNING`
   - [Seed 7](https://www.kaggle.com/code/franckyoteu/exp-lsh-pipeline-seed-7) — `RUNNING`
   - [Seed 123](https://www.kaggle.com/code/franckyoteu/exp-lsh-pipeline-seed-123) — `RUNNING`
   - [Seed 2024](https://www.kaggle.com/code/franckyoteu/exp-lsh-pipeline-seed-2024) — `RUNNING`
   - [Seed 99](https://www.kaggle.com/code/franckyoteu/exp-lsh-pipeline-seed-99) — `RUNNING`
3. **Execution Scope per Kernel:**
   - Validation calibration on Heart Disease and ChestX-ray14 (10k, 50k, full).
   - Selection & parameter freezing according to Protocol Section 8.
   - Ground truth caching (`linear_gt_*.npz`).
   - Test benchmark across all 3 hash seeds (`[11, 29, 43]`) with 1 warmup + 5 measured repetitions per query.
   - CSV artifact generation into `/kaggle/working/`.

---

## 2026-08-16 — Phase 2: Results Analysis & Artifacts Generation

**Agent:** Antigravity (conversation `97f21a6c-69e0-44e7-9e47-474511cf7f8d`)

1. **Artifacts Retrieval:** All CSVs, JSONs, and `.npz` ground truth files downloaded from the 5 Kaggle sessions.
2. **Master Aggregation:** Merged into `benchmark_quality.csv`, `benchmark_timing.csv`, `benchmark_candidates.csv`, `benchmark_memory.csv`, and `lsh_calibration.csv`.
3. **Statistical Analysis:** Wilcoxon paired tests and bootstrap confidence intervals computed across all splits (`significance_tests.csv`).
4. **Summary & Registry:** Saved system environment to `registry.json` and high-level outcomes to `summary.json`.
5. **8 Figures Generated:** All saved to `exp_lsh/results/figures/`:
   - `fig01_latency.png` — Query latency comparison (Linear vs SimHash vs E2LSH).
   - `fig02_recall.png` — Recall@1/@5/@10 curves.
   - `fig03_candidates.png` — Base fraction examined (candidate filtering efficiency).
   - `fig04_build_time.png` — Index build time across base sizes.
   - `fig05_memory.png` — Memory footprint comparison.
   - `fig06_recall_latency_tradeoff.png` — Latency vs Recall trade-off scatter plot.
   - `fig07_scaling.png` — ChestX-ray14 scaling behavior (10k, 50k, 72k).
   - `fig08_hash_seed_variance.png` — Stability of E2LSH across random projection seeds.

6. **Key Scientific Findings:**
   - **ChestX-ray14 ($d=1024$):** Both SimHash and E2LSH successfully preserved retrieval quality ($\text{Recall@10} \approx 96.8\% - 98.5\%$, satisfying the $\ge 0.95$ threshold, and classification macro-F1 preserved with $\Delta \text{F1} \le 0.01$). Outcome classified as **"Qualité préservée sans gain de temps"** due to Python dictionary traversal vs highly vectorized BLAS matrix multiplications in single-core NumPy.
   - **Heart Disease ($d=13$):** Low-dimensional small-base retrieval is dominated by linear scan; random projections in low dimensions fail to partition effectively (**"Échec global"** as expected theoretically).

---

## Status: Phase 1 Complete ✅ — follow-up in exp_falconn

Phase 1 (SimHash + E2LSH, L2 arm) is complete. Key findings:

- **ChestX-ray14 (d=1024):** both LSH methods preserved retrieval quality
  (Recall@10 ≈ 0.968–0.985 ≥ 0.95, Δmacro-F1 ≤ 0.01) but showed **no time gain**
  vs the BLAS-vectorised linear scan in single-core NumPy (dict traversal vs
  `np.linalg.norm` broadcast).
- **Heart Disease (d=13):** low-dimensional small-base retrieval is dominated by
  linear scan; random projections fail to partition — **"Échec global"** as expected.

### Follow-up run

The cross-polytope LSH (the theoretically *optimal* angular LSH, FALCONN-style,
implemented from scratch in pure NumPy) is evaluated on the **normalised cosine
arm** in **`../exp_falconn/`**. SimHash/E2LSH are not re-run; this folder's
phase-1 results are reused for the final cross-method comparison.

---

*Last updated: 2026-08-16*
