"""Cellules du notebook NB03 (benchmark des index seuls — T1.1 à T1.6)."""

CELLS = [
    ("md", """\
# NB03 — Benchmark des index seuls

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Benchmark des structures d'index sans maintenance, sur les **3 datasets × 6 seeds** :

- **T1.1** Baseline : scan linéaire BLAS (einsum/matvec) — `LinearScanIndex`.
- **T1.2** KD-tree (sklearn) — `KDTreeIndex`.
- **T1.3** LSH équitable (reclassement einsum/BLAS, pas `np.linalg.norm`) + calibration
  bits/tables.
- **T1.4** HeCBR (Diabetes-130 uniquement) : codes pré-entraînés, distance de Hamming,
  recherche restreinte au train.
- **T1.5** Temps + qualité sur k={1,5,10} × 6 seeds (warm-up + médiane, T0.5).
- **T1.6** Empreinte mémoire (pic RSS par sous-processus + tracemalloc par requête).

**Protocole (a priori, reproductible)** : requêtes par dataset = 60 (heart, tout le test) /
500 (diabetes130) / 300 (chestxray14), échantillonnées par seed ; build mesuré 1× par
(dataset, méthode) ; checkpoint CSV incrémental — une ligne déjà écrite n'est jamais
réécrite (reprise après interruption).

Sorties : `benchmark_timing_quality.csv`, `benchmark_memory.csv`, `lsh_calibration.csv`,
`nb03_summary.json`. Ce notebook est conçu pour Kaggle (lourd : KD-tree 1024-d) ;
variables locales de réduction : `NB03_SEEDS`, `NB03_QUERY_CAP`.
"""),

    ("code", """\
import sys
from pathlib import Path

_candidates = [Path.cwd() / "exp2"]
if Path("/kaggle/input").exists():
    _candidates += sorted(Path("/kaggle/input").glob("*/"))
EXP2 = next((p for p in _candidates if (p / "src").exists() and (p / "config").exists()), None)
assert EXP2 is not None, "Dossier exp2 introuvable (local: cwd/exp2, Kaggle: /kaggle/input/*)"
sys.path.insert(0, str(EXP2))
print("EXP2 =", EXP2)
"""),

    ("code", """\
import json

import numpy as np
import pandas as pd

from config import benchmark as bench_cfg
from config import datasets as ds_cfg
from config import run as run_cfg, seeds

run_dir = run_cfg.get_run_dir()
print("Run courant :", run_dir)
print()
print("Requêtes par dataset :", bench_cfg.N_QUERIES_BY_DATASET)
print(f"Timing : warm-up={bench_cfg.QUERY_WARMUP}, répétitions={bench_cfg.QUERY_REPETITIONS}")
print("Seeds benchmark :", seeds.ALL_SEEDS)
print("k :", seeds.K_VALUES)
print("Params LSH par défaut :", bench_cfg.LSH_PARAMS_BY_DATASET)
"""),

    ("md", """\
## T1.3 — Calibration LSH (bits × tables, rappel ≥ 0.95)

Sur 30 requêtes du train (seed 42), grille bits × tables {8,16,32} ; la combinaison
retenue (rappel ≥ 0.95, sinon meilleur rappel) devient le paramètre du benchmark.
"""),

    ("code", """\
from src.benchmark import benchmark_indexes as bx
from src.datasets import adapters

chosen_params = {}
calib_records = []
for key in ds_cfg.ORDER:
    print(f"=== calibration {key} ===")
    X_train, _, _, _, _, _ = adapters.get_split(key, seed=42)
    params, records = bx.calibrate_lsh(key, X_train, progress=print)
    chosen_params[key] = params
    calib_records.append(records)
    bench_cfg.LSH_PARAMS_BY_DATASET[key] = params  # appliqué au benchmark
    print(f"  -> retenu : {params}")

calib_df = pd.concat(calib_records, ignore_index=True)
calib_df.to_csv(run_dir / "lsh_calibration.csv", index=False)
print("\\nÉcrit :", run_dir / "lsh_calibration.csv")
print("Params LSH appliqués au benchmark :", bench_cfg.LSH_PARAMS_BY_DATASET)
"""),

    ("md", """\
## T1.1 / T1.2 / T1.5 — Benchmark exact / KD-tree / LSH

Pour chaque dataset × seed : split stratifié, construction, requêtes, puis métriques
(rangement + rappel des voisins exacts + classification) et temps (médiane). Checkpoint
CSV incrémental.
"""),

    ("code", """\
for key in ds_cfg.ORDER:
    print(f"\\n########## BENCHMARK {key} ##########")
    bx.benchmark_dataset(key, run_dir, progress=print)

print("\\nBenchmark terminé.")
print("Aperçu :", run_dir / "benchmark_timing_quality.csv")
pd.read_csv(run_dir / "benchmark_timing_quality.csv").head(8)
"""),

    ("md", """\
## T1.4 — HeCBR (Diabetes-130, codes pré-entraînés)

Distance de Hamming, recherche restreinte aux cas `train` du parquet. Le rappel des
voisins exacts euclidiens n'a pas de sens ici (espace de codes différent) : qualité de
classification uniquement.
"""),

    ("code", """\
bx.benchmark_hecbr(run_dir, progress=print)
pd.read_csv(run_dir / "benchmark_timing_quality.csv").query("method == 'hecbr'")
"""),

    ("md", """\
## T1.6 — Empreinte mémoire (pic RSS index + allocations par requête)

Chaque (dataset, méthode) est mesuré dans un **sous-processus frais** : le pic RSS
(`ru_maxrss`) après construction est alors propre à la méthode ; les allocations d'une
requête sont mesurées par `tracemalloc`.
"""),

    ("code", """\
for key in ds_cfg.ORDER:
    print(f"=== mémoire {key} ===")
    bx.benchmark_memory(key, run_dir, progress=print)

print("\\nÉcrit :", run_dir / "benchmark_memory.csv")
pd.read_csv(run_dir / "benchmark_memory.csv")
"""),

    ("code", """\
from datetime import datetime

res = pd.read_csv(run_dir / "benchmark_timing_quality.csv")
mem = pd.read_csv(run_dir / "benchmark_memory.csv")
summary = {
    "notebook": "NB03",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "protocol": {
        "n_queries_by_dataset": bench_cfg.N_QUERIES_BY_DATASET,
        "query_warmup": bench_cfg.QUERY_WARMUP,
        "query_repetitions": bench_cfg.QUERY_REPETITIONS,
        "seeds": seeds.ALL_SEEDS,
        "k_values": seeds.K_VALUES,
        "lsh_params_chosen": chosen_params,
        "hecbr_codes": bench_cfg.HECBR_CODES_PATH,
    },
    "result_rows": int(len(res)),
    "memory_rows": int(len(mem)),
    "datasets_done": sorted(res["dataset"].unique().tolist()),
    "methods_done": sorted(res["method"].unique().tolist()),
    "seeds_done": sorted(res["seed"].astype(str).unique().tolist()),
}
(run_dir / "nb03_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb03_summary.json")
print("Lignes de résultats :", len(res), "| mémoire :", len(mem))
print("Méthodes :", summary["methods_done"], "| seeds :", summary["seeds_done"])
"""),

    ("md", """\
## Validation

- [x] T1.1 — Baseline scan linéaire BLAS (référence exacte par requête)
- [x] T1.2 — KD-tree sklearn (build + requêtes)
- [x] T1.3 — LSH équitable einsum/BLAS + calibration bits/tables (rappel ≥ 0.95)
- [x] T1.4 — HeCBR : codes pré-entraînés, Hamming, split fixe, train restreint
- [x] T1.5 — temps + qualité sur k={1,5,10} × 6 seeds (checkpoint CSV incrémental)
- [x] T1.6 — empreinte mémoire par sous-processus (pic RSS + tracemalloc par requête)
- [x] Sorties : `benchmark_timing_quality.csv`, `benchmark_memory.csv`,
      `lsh_calibration.csv`, `nb03_summary.json`

**Étape suivante** : NB04 — figures benchmark (FIG-R1, FIG-R2) depuis ces CSV.""")
]
