# Task Board — exp_falconn (cross-polytope run)

> ✅ **Phase 2 terminee — toutes les taches done. Reste: redaction (demo, papier).**

> Tasks are grouped by status. Update this file whenever a task changes state.
> This run evaluates **only** the linear baseline and the **CrossPolytope LSH**
> (FALCONN-style, pure NumPy) on the **normalised cosine arm**.
> SimHash / E2LSH are *not* re-run here; phase-1 results from `exp_lsh/` are
> reused for the final comparison.
> Format each task as: `- [ ] Task description — *owner / date*`

---

## ✅ Completed

- [x] Scaffold `exp_falconn/` (mirror of `exp_lsh/`) — config, src, tests, requirements.txt — *big-pickle, 2026-08-16*
- [x] Read and apply `draft/protocol.md` §5 (cosine arm: L2-normalise heart + chest, rank by L2 on the unit sphere) — *big-pickle, 2026-08-16*
- [x] Write `config/metrics.py` with `PRIMARY_METRIC = "cosine"`, `CHEST_COSINE_ARM_ENABLED = True` — *big-pickle, 2026-08-16*
- [x] Write `src/indexes/cross_polytope.py` — FALCONN-style CrossPolytope (rotation `H·D₃·H·D₂·H·D₁`, vectorised FHT, hash `2·idx + sign`, multi-probe, `last_cp_dimensions`, memory estimate) — *big-pickle, 2026-08-16*
- [x] Write `src/indexes/linear_exact.py` — unchanged BLAS-optimised baseline — *big-pickle, 2026-08-16*
- [x] Write `src/calibration/grid_search.py::calibrate_cross_polytope` — val-only grid over num_hashes/num_tables/num_rotations/num_probes, `_select_best` tie-break (recall ≥ 0.95, ΔF1 ≤ 0.01, median latency → candidate fraction) — *big-pickle, 2026-08-16*
- [x] Add `normalise_cosine=True` option to `src/datasets/heart_disease.py` (L2-normalise after StandardScaler) — *big-pickle, 2026-08-16*
- [x] Performance refactor: transposed `(d, n)` FHT layout (~3× faster), batched query rotation `_rotate_query_all`, vectorised multi-probe — *big-pickle, 2026-08-16*
- [x] Match FALCONN defaults (verified against `falconn_src`): `num_rotations=2`, K from `number_of_hash_bits`, `l=10` heuristic — *big-pickle, 2026-08-16*
- [x] Write `tests/test_cross_polytope.py` and run `pytest tests/` (21/21 passed) — *big-pickle, 2026-08-16*
- [x] Generate reference notebooks `00_data_checks` → `04_analysis` (cross-polytope only) — *big-pickle, 2026-08-16*
- [x] Upload package `franckyoteu/exp-falconn-code` to Kaggle (dir_mode=zip) — *big-pickle, 2026-08-16*
- [x] Push 5 pipeline kernels (actual slugs `exp-falconn-pipeline-seed-{42,7,123,2024,99}`) — *big-pickle, 2026-08-16*
- [x] Run 5 kernels to completion (zero errors, ~48 min on Kaggle) and download all 50 output files — *big-pickle, 2026-08-16*
- [x] Paired Wilcoxon tests (linear vs CP) + bootstrap CIs → `significance_tests.csv` — *big-pickle, 2026-08-16*
- [x] Generate the 8 protocol figures → `results/figures/` — *big-pickle, 2026-08-16*
- [x] Write `summary.json` (protocol §14) and `outcomes.csv` (protocol §12) — *big-pickle, 2026-08-16*
- [x] Classify outcomes: all 4 tasks = **quality_ok_no_speedup** — *big-pickle, 2026-08-16*
- [x] Patch `cross_polytope.py` with `_D_all_query` precomputed contigu — non-regression 6/6 tests (150 queries × 4 configs) — *big-pickle, 2026-08-16*
- [x] Audit anti-patterns (.T, reshape) in cross_polytope, e2lsh, simhash, grid_search — clean (false positives only) — *big-pickle, 2026-08-16*
- [x] Relaunch 5 patched Kaggle kernels, download results (5/5 complete, 38 min, zero errors) — *big-pickle, 2026-08-16*
- [x] Update DECISIONS_AND_RESULTS.md §7.2, 7.3, 8.1 + lsh_sota_review.md §7 with patched figures — *big-pickle, 2026-08-16*
- [x] Add methodology principle (no BLAS borrowing, only remove accidental handicaps) to progress.md — *big-pickle, 2026-08-16*

---

## 📋 To Do

### Data acquisition
- [ ] Obtain Heart Disease dataset CSV and place at expected path — *unassigned*
- [ ] Obtain ChestX-ray14 embeddings (`.npy`, shape `(112120, 1024)`) and labels — *unassigned*

### Reporting
- [x] Write `registry.json` (protocol §9.3) — *big-pickle, 2026-08-16*
- [x] Write `docs/lsh_sota_review.md` (cross-polytope vs phase-1 SOTA comparison) — *big-pickle, 2026-08-16*

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
- Re-running SimHash / E2LSH (reuse `exp_lsh/` phase-1 results instead)

---

*Last updated: 2026-08-16*
