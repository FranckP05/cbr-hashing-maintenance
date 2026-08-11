"""Cellules du notebook NB00 (configuration & registre)."""

CELLS = [
    ("md", """\
# NB00 — Configuration & registre

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Ce notebook :
1. déclare les chemins (local vs Kaggle), seeds et datasets (T0.1) ;
2. crée le **run horodaté** et son **registre JSON** (T0.2) ;
3. vérifie que l'**adapter générique** charge, splitte et prétraite chaque dataset
   (scaler ajusté sur le train uniquement — règle §1.3).

Les notebooks suivants lisent le pointeur `current_run.json` pour savoir où écrire.
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
from config import datasets as ds_cfg
from config import paths, run as run_cfg, seeds

print("Environnement :", paths.ENV)
print("DATA_ROOT     :", paths.DATA_ROOT)
print("RUN_ROOT      :", paths.RUN_ROOT)
print("Seeds         :", seeds.SEEDS, "| sampling:", seeds.SAMPLING_SEED)
print("Test fraction :", seeds.TEST_FRACTION, "| k values:", seeds.K_VALUES)
print()
for key in ds_cfg.ORDER:
    cfg = ds_cfg.DATASETS[key]
    print(f"- {key:12s} {cfg['label']:32s} dim={cfg['n_features']:5d}  scaling={cfg['scaling']}")
    print(f"    source: {ds_cfg.source(key)}")
"""),

    ("code", """\
from datetime import datetime

run_dir = run_cfg.new_run(tag="benchmark")
(run_dir / "figures").mkdir(exist_ok=True)
print("Run créé :", run_dir)

registry = {
    "experiment": "exp2",
    "plan_doc": "docs/plan_experimentation_reproductible.md",
    "env": paths.ENV,
    "created": datetime.now().isoformat(timespec="seconds"),
    "seeds": seeds.SEEDS,
    "sampling_seed": seeds.SAMPLING_SEED,
    "test_fraction": seeds.TEST_FRACTION,
    "k_values": seeds.K_VALUES,
    "datasets": {
        k: {
            "label": ds_cfg.DATASETS[k]["label"],
            "description": ds_cfg.DATASETS[k]["description"],
            "scaling": ds_cfg.DATASETS[k]["scaling"],
            "n_features": ds_cfg.DATASETS[k]["n_features"],
            "source": ds_cfg.source(k),
        }
        for k in ds_cfg.ORDER
    },
}
run_cfg.write_registry(run_dir, registry)
print("Registre écrit :", run_dir / "registry.json")
"""),

    ("code", """\
import numpy as np
import pandas as pd

from src.datasets import adapters

rows = []
for key in ds_cfg.ORDER:
    X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key)
    rows.append({
        "dataset": key,
        "n": int(len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "d": int(X_train.shape[1]),
        "dtype": str(X_train.dtype),
        "pos_ratio_train": round(float(np.mean(y_train)), 4),
    })
    print(f"OK {key:12s} X_train={X_train.shape}  X_test={X_test.shape}  "
          f"pos_ratio_train={np.mean(y_train):.4f}")

profile = pd.DataFrame(rows)
profile.to_csv(run_dir / "data_profile.csv", index=False)
print("\\nProfil écrit :", run_dir / "data_profile.csv")
"""),

    ("md", """\
## Validation

- [x] Chemin `exp2` résolu (local ou Kaggle)
- [x] Seeds fixes : `{42, 7, 123, 2024, 99}` + sampling `1337`
- [x] Run horodaté + registre JSON (T0.2)
- [x] Adapter générique : split stratifié par seed, prétraitement sur train seul

**Étape suivante** : NB01 — exploration des données et figures de présentation
(FIG-D1..D6)."""),
]
