"""Gestion des runs : dossier versionné + registre (T0.2).

Chaque run est un dossier horodaté sous exp2/results/. Le notebook 00 crée le run
et écrit un pointeur (current_run.json) que les notebooks suivants lisent pour
savoir où écrire leurs CSV/figures.
"""

import json
import time
from pathlib import Path

from config.paths import RUN_ROOT

CURRENT_RUN_FILE = RUN_ROOT / "current_run.json"


def new_run(tag=None):
    """Crée un nouveau run horodaté et le déclare comme run courant."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}" + (f"_{tag}" if tag else "")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_RUN_FILE.write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}))
    return run_dir


def get_run_dir():
    """Renvoie le run courant ; en crée un si aucun n'existe."""
    if CURRENT_RUN_FILE.exists():
        return Path(json.loads(CURRENT_RUN_FILE.read_text())["run_dir"])
    return new_run()


def write_registry(run_dir, data):
    """Écrit le registre JSON du run (métadonnées complètes)."""
    (run_dir / "registry.json").write_text(json.dumps(data, indent=2, default=str))
