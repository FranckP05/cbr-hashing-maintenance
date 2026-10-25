"""Construction / reconstruction des index — réutilise exp2/src/indexing/.

Les trois index (balayage linéaire, KD-Tree, LSH) sont construits sur les vecteurs bruts
du train. Pour Diabetes-130, un 4e index HeCBR (codes binaires pré-entraînés, distance de
Hamming, restreint au split train du parquet) complète la démo.
"""

from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.linear_scan import LinearScanIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex

from demo_config import DATASETS, HECBR_CODES_PATH


def build_dataset_indexes(key, vectors):
    """Construit les index du dataset (appelé à la 1re sélection et au reset)."""
    indexes = {
        "linear": LinearScanIndex(),
        "kdtree": KDTreeIndex(),
        "lsh": RandomProjectionLSHIndex(**DATASETS[key]["lsh_params"]),
    }
    for index in indexes.values():
        index.build(vectors)

    if key == "diabetes130":
        from src.indexing.hecbr_index import HeCBRIndex

        hecbr = HeCBRIndex(HECBR_CODES_PATH)
        hecbr.build(None, restrict_split="train")
        indexes["hecbr"] = hecbr

    return indexes


def reset_dataset(key):
    """Réinitialise le dataset actif : index reconstruits depuis les vecteurs bruts."""
    from state import demo_state

    vectors = demo_state.pristine_train_vectors[key]
    demo_state.active_indexes[key] = build_dataset_indexes(key, vectors)
    demo_state.base_size[key] = vectors.shape[0]
    demo_state.current_case = None
    demo_state.current_case_meta = {}
