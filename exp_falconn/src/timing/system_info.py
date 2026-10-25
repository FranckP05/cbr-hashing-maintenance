"""
System and environment information recorder (section 9.3).

Records Python, NumPy, SciPy, scikit-learn, FAISS versions,
CPU model, RAM, and BLAS/OpenMP thread count.
Saved to the run registry (registry.json).
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Dict, Any


def collect() -> Dict[str, Any]:
    """Collect system and library version information.

    Returns:
        Dict suitable for inclusion in registry.json.
    """
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }

    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
        info["numpy_blas"] = np.__config__.blas_opt_info.get("libraries", [])
    except Exception:
        info["numpy_version"] = "unavailable"

    # SciPy
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        info["scipy_version"] = "not installed"

    # scikit-learn
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = "not installed"

    # FAISS (optional)
    try:
        import faiss
        info["faiss_version"] = getattr(faiss, "__version__", "installed")
    except ImportError:
        info["faiss_version"] = "not installed"

    # OpenMP / BLAS threads
    info["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "not set")
    info["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS", "not set")

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        info["ram_gb"] = "psutil not installed"

    return info
