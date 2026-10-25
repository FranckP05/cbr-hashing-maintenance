"""État mono-session de la démo (voir docs/demo.md — section État)."""


class DemoState:
    """État global partagé par tous les visiteurs (mono-session assumée).

    Par dataset déjà sélectionné : vecteurs bruts du train/test (immuables), index actifs
    (mutables via add_case()), taille de base. current_dataset pointe le dataset actif ;
    current_case contient le dernier cas généré/saisi en attente d'analyse.
    """

    def __init__(self):
        self.pristine_train_vectors = {}  # key -> np.ndarray (jamais modifié)
        self.pristine_test_vectors = {}   # key -> np.ndarray (génération auto)
        self.pristine_test_targets = {}   # key -> np.ndarray (libellé du cas auto)
        self.train_encounter_ids = {}     # key -> np.ndarray aligné sur X_train (HeCBR D130)
        self.test_encounter_ids = {}      # key -> np.ndarray aligné sur X_test
        self.active_indexes = {}          # key -> {"linear","kdtree","lsh"} + "hecbr" si D130
        self.base_size = {}               # key -> int, évolue avec add_case()
        self.current_dataset = None
        self.current_case = None          # vecteur en attente d'analyse
        self.current_case_meta = {}       # {"source","encounter_id","target"}
        self.scaler = {}                  # key -> scaler ajusté sur train (saisie manuelle heart)
        self.feature_ranges = {}          # key -> {col: (min, max)} train (guidage saisie)
        self.hecbr_codes = None           # df codes D130 (lookup par encounter_id)
        self.last_results = None          # résultats de la dernière analyse (ré-affichage)
        self.last_before = None
        self.last_after = None
        self.manual_mode = False          # True si l'user a cliqué "Entrer à la main"
        self.manual_errors = {}           # erreurs de validation formulaire manual
        self.manual_values = {}           # valeurs du formulaire avant validation
        self.cbm_applied = {}             # key -> bool (effet global maintenance affiché)


demo_state = DemoState()
