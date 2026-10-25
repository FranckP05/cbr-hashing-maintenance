"""Point d'entrée principal : Pipeline complet de bout en bout."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline complet LSH Camelyon16.")
    p.add_argument(
        "--k", type=int, default=10,
        help="Nombre de voisins k (défaut : 10).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Ne pas générer les figures.",
    )
    p.add_argument(
        "--skip-calibration", action="store_true",
        help="Sauter la calibration (utilise un best_params_simhash.json existant).",
    )
    return p.parse_args()


def run_step(script: Path, extra_args: list) -> None:
    """Exécute un sous-script Python et lève une exception en cas d'erreur."""
    cmd = [sys.executable, str(script)] + extra_args
    logger.info("Lancement : %s", " ".join(str(c) for c in cmd))
    t0  = time.perf_counter()
    ret = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        logger.error("ERREUR dans %s (code %d)", script.name, ret.returncode)
        sys.exit(ret.returncode)
    logger.info("%s terminé en %.1f s.", script.name, elapsed)


def main() -> None:
    args     = parse_args()
    scripts  = Path(__file__).resolve().parent
    t_start  = time.perf_counter()

    extra = [f"--k={args.k}"]
    if args.no_plots:
        extra.append("--no-plots")

    # Étape 1 — Calibration
    if not args.skip_calibration:
        logger.info("=" * 60)
        logger.info("ÉTAPE 1/2 — Calibration SimHash (validation)")
        logger.info("=" * 60)
        run_step(scripts / "run_calibration.py", extra)
    else:
        logger.info("Calibration ignorée (--skip-calibration).")

    # Étape 2 — Benchmark
    logger.info("=" * 60)
    logger.info("ÉTAPE 2/2 — Benchmark comparatif final (test, multi-seeds)")
    logger.info("=" * 60)
    run_step(scripts / "run_benchmark.py", extra)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complet terminé en %.1f s (%.1f min).", elapsed, elapsed / 60)
    logger.info("Résultats : lsh_camelyon/results/raw/")
    logger.info("Figures   : lsh_camelyon/results/figures/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
