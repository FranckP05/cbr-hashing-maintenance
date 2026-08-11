"""Protocole de maintenance (T3) — constantes a priori.

Documenté dans NB07 :
- **Rétention (T3.1)** : seuil θ = 25e percentile des distances 1-NN du train (échantillon
  de RETENTION_THRESHOLD_SAMPLE cas, a priori) ; flux = cas du test échantillonnés par
  seed ; la base croît par rétention (add_case, LSH vs référence exacte) ; mesures :
  accord cas par cas, taux de réduction, taille finale, coût par cas, taille du bassin
  de candidats LSH (collisions).
- **Condensation (T3.2)** : taux cibles 20 % / 50 % du train ; stratégies sur le MÊME
  sous-échantillon (premiers COND_SUBSAMPLE cas du train du seed) :
    random (baseline), cbm (couverture relative, exacte), bucket (hachage : num_bits a
    priori par dataset, 1 table de référence, représentant = cas le plus proche du
    centroïde du compartiment), cnn (exacte, taille naturelle ; si taille naturelle >
    cible, on garde les premiers pivots de l'ordre de condensation).
- **Post-maintenance (T3.3/T3.4)** : qualité 1-NN (accuracy, macro-F1, balanced
  accuracy) + temps de requête (k=10, linear/kdtree/lsh) sur base condensée vs brute
  (même sous-échantillon) ; build mesuré pour chaque condensation.
"""

RETENTION_NN_PERCENTILE = 25
RETENTION_THRESHOLD_SAMPLE = 2000
RETENTION_LSH_K = 1

COND_SUBSAMPLE = 10_000
COND_RATES = [0.2, 0.5]
COND_STRATEGIES = ["random", "cbm", "bucket", "cnn"]

# num_bits pour la condensation par bucket (1 table) : assez de compartiments pour
# atteindre le taux cible le plus bas (20 % du sous-échantillon)
BUCKET_NUM_BITS_BY_DATASET = {"heart": 6, "diabetes130": 11, "chestxray14": 11}

POST_QUERY_K = 10
