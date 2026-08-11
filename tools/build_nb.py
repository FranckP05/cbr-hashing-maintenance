"""Outils de production des notebooks exp2.

build_nb.py : construit un .ipynb à partir d'un module de cellules Python.
exec_nb.py  : exécute en séquence les cellules code d'un .ipynb (vérification).
"""

import importlib.util
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build(cells, output_path):
    """cells: liste de tuples (kind, source) avec kind in {'md', 'code'}."""
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    for kind, source in cells:
        cell = new_markdown_cell(source) if kind == "md" else new_code_cell(source)
        nb.cells.append(cell)
    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    print(f"Écrit {output_path} ({len(cells)} cellules)")


def load_cells(module_path):
    spec = importlib.util.spec_from_file_location("cells_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cells_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.CELLS


def main():
    module_path, output_path = sys.argv[1], sys.argv[2]
    build(load_cells(module_path), output_path)


if __name__ == "__main__":
    main()
