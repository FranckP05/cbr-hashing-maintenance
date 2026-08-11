"""Exécute en séquence les cellules code d'un notebook .ipynb.

Usage : python tools/exec_nb.py exp2/notebooks/NB00_config_registre.ipynb
Vérification locale : sorties écrites dans le run, figures en backend Agg.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import nbformat  # noqa: E402


def main(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        first_line = cell.source.strip().splitlines()[0][:70]
        print(f"\n=== cellule {i}: {first_line}")
        exec(compile(cell.source, f"{nb_path}#cell{i}", "exec"), ns)
    print("\nOK: toutes les cellules exécutées.")


if __name__ == "__main__":
    main(sys.argv[1])
