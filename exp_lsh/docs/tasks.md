# Task Board — exp_lsh

> **Phase 1 (this folder) is COMPLETE ✅.** SimHash and E2LSH were evaluated;
> neither beats linear on heart; on chest quality was preserved (Recall@10 ≈ 0.97–0.98,
> ΔF1 ≤ 0.01) but without a time gain in the NumPy prototype.
> The follow-up **Phase 2 (CrossPolytope LSH, cosine arm)** lives in
> **`../exp_falconn/`** — see its `docs/tasks.md` / `docs/progress.md`.

> Tasks are grouped by status. Update this file whenever a task changes state.
> Format each task as: `- [ ] Task description — *owner / date*`

---

## 🔵 In Progress

*(none currently)*

---

## ✅ Completed

- [x] Read and parse `../draft/protocol.md` — *Antigravity, 2026-08-16*
- [x] Create full project directory structure (`exp_lsh/`) — *Antigravity, 2026-08-16*
- [x] Write `README.md` — *Antigravity, 2026-08-16*
- [x] Write `requirements.txt` — *Antigravity, 2026-08-16*
- [x] Write `config/datasets.py` — DatasetConfig for Heart Disease & ChestX-ray14 — *Antigravity, 2026-08-16*
- [x] Write `config/protocol.py` — quality thresholds, timing constants, split fractions — *Antigravity, 2026-08-16*
- [x] Write `config/seeds.py` — SPLIT_SEEDS, HASH_SEEDS, SEED_SAMPLING — *Antigravity, 2026-08-16*
- [x] Write `config/metrics.py` — RECALL_KS, PRIMARY_METRIC, CBR_VOTE_RULE — *Antigravity, 2026-08-16*
- [x] Write `src/datasets/heart_disease.py` — loader + StandardScaler pipeline — *Antigravity, 2026-08-16*
- [x] Write `src/datasets/chest_xray14.py` — embedding loader + sub-base sampling — *Antigravity, 2026-08-16*
- [x] Write `src/indexes/linear_exact.py` — BLAS-optimised exhaustive L2 search — *Antigravity, 2026-08-16*
- [x] Write `src/indexes/simhash.py` — random-hyperplane ablation — *Antigravity, 2026-08-16*
- [x] Write `src/indexes/e2lsh.py` — E2LSH p-stable with multi-probe — *Antigravity, 2026-08-16*
- [x] Write `src/calibration/grid_search.py` — validation-set calibration — *Antigravity, 2026-08-16*
- [x] Write `src/evaluation/metrics.py` — Recall@k, nDCG, macro-F1, AUC, CBR — *Antigravity, 2026-08-16*
- [x] Write `src/timing/harness.py` — timed_query() with warm-up + median — *Antigravity, 2026-08-16*
- [x] Write `src/timing/system_info.py` — environment/version recorder — *Antigravity, 2026-08-16*
- [x] Write `tests/test_splits.py` — split overlap + NaN checks — *Antigravity, 2026-08-16*
- [x] Write `tests/test_linear.py` — LinearExact correctness — *Antigravity, 2026-08-16*
- [x] Write `tests/test_e2lsh.py` — E2LSH correctness + recall smoke test — *Antigravity, 2026-08-16*
- [x] Write `notebooks/00_data_checks.ipynb` — full data validation, split checks, profile CSVs — *Antigravity, 2026-08-16*
- [x] Write `notebooks/01_calibration.ipynb` — vectorized E2LSH/SimHash grid search, val-only calibration, JSON output — *Antigravity, 2026-08-16*
- [x] Write `notebooks/02_benchmark.ipynb` — ground-truth caching, full benchmark (linear/SimHash/E2LSH), 4 output CSVs — *Antigravity, 2026-08-16*
- [x] Write `notebooks/03_scaling.ipynb` — ChestX 10k/50k/full scaling figures, speedup table — *Antigravity, 2026-08-16*
- [x] Write `notebooks/04_analysis.ipynb` — Wilcoxon tests, bootstrap CIs, 8 protocol figures, summary.json — *Antigravity, 2026-08-16*
- [x] Create `docs/progress.md` and `docs/tasks.md` — *Antigravity, 2026-08-16*

---

## 📋 To Do

### Data acquisition
- [ ] Obtain Heart Disease dataset CSV and place at expected path — *unassigned*
- [ ] Obtain ChestX-ray14 embeddings (`.npy`, shape `(112120, 1024)`) and labels — *unassigned*
- [ ] Document final data paths in `config/datasets.py` — *unassigned*

- [x] Run `pytest tests/` and confirm all tests pass (16/16 passed) — *Antigravity, 2026-08-16*
- [x] Phase 0 data validation: no NaN/Inf, no split overlap across all 5 seeds (`dataset_profile.csv`, `split_profile.csv` generated) — *Antigravity, 2026-08-16*

### Calibration & Benchmarking — Kaggle Parallel Dispatch (Phase 1)
- [x] Create and push code package `franckyoteu/exp-lsh-code` to Kaggle — *Antigravity, 2026-08-16*
- [x] Generate 5 end-to-end pipeline notebooks & push kernels to Kaggle:
  - `franckyoteu/exp-lsh-seed42`
  - `franckyoteu/exp-lsh-seed7`
  - `franckyoteu/exp-lsh-seed123`
  - `franckyoteu/exp-lsh-seed2024`
  - `franckyoteu/exp-lsh-seed99` — *Antigravity, 2026-08-16*
- [x] Run parallel sessions to completion and download output CSVs (`lsh_calibration_seed*.csv`, `benchmark_*_seed*.csv`) — *Antigravity, 2026-08-16*

### Statistical analysis (§11)
- [x] Paired Wilcoxon tests (linear vs. E2LSH) for latency, Recall@10, macro-F1 — *Antigravity, 2026-08-16*
- [x] Bootstrap confidence intervals — *Antigravity, 2026-08-16*
- [x] Separate variance: split seeds vs. hash seeds — *Antigravity, 2026-08-16*
- [x] Save `significance_tests.csv` — *Antigravity, 2026-08-16*

### Figures (§14)
- [x] Fig 1: latency linear vs SimHash vs E2LSH (`fig01_latency.png`) — *Antigravity, 2026-08-16*
- [x] Fig 2: Recall@1/@5/@10 (`fig02_recall.png`) — *Antigravity, 2026-08-16*
- [x] Fig 3: fraction of candidates (`fig03_candidates.png`) — *Antigravity, 2026-08-16*
- [x] Fig 4: build time (`fig04_build_time.png`) — *Antigravity, 2026-08-16*
- [x] Fig 5: memory (`fig05_memory.png`) — *Antigravity, 2026-08-16*
- [x] Fig 6: recall-latency trade-off (`fig06_recall_latency_tradeoff.png`) — *Antigravity, 2026-08-16*
- [x] Fig 7: scaling Chest 10k/50k/full (`fig07_scaling.png`) — *Antigravity, 2026-08-16*
- [x] Fig 8: variance across hash seeds (`fig08_hash_seed_variance.png`) — *Antigravity, 2026-08-16*

### Reporting
- [x] Write `summary.json` (protocol §14) — *Antigravity, 2026-08-16*
- [x] Write `registry.json` (protocol §9.3) — *Antigravity, 2026-08-16*
- [x] Classify result outcomes (Protocol Section 12) — *Antigravity, 2026-08-16*

---

## 🚫 Explicitly Out of Scope (protocol §15)

> The following will **never** be added to this experiment:

- HeCBR
- Case retention / condensation / CBM / KDTree
- CNN
- Bucket-condensation
- Supervised hash learning
- Parameter selection on test data
- Any artificial modification of the linear baseline

---

*Last updated: 2026-08-16*
