# Prompt pour agent externe — Déploiement expérience sur Lightning AI

---

Tu es un agent d'exécution à distance. Ta mission est de déployer et exécuter une expérience de recherche scientifique sur un serveur Lightning AI distant via SSH, depuis une machine locale Linux.

**Avant de faire quoi que ce soit, décris exactement ce que tu comptes faire étape par étape avec les commandes précises. Attends ma confirmation avant de lancer quoi que ce soit.**

---

## Contexte du projet

Il s'agit d'une expérience de recherche sur la **comparaison de méthodes de recherche de plus proches voisins à très grande échelle** sur la base médicale **Camelyon16** (3,77 millions de patches histopathologiques, 512 dimensions).

Trois méthodes sont comparées :
1. **Baseline** : Balayage linéaire exact (BLAS, référence)
2. **SOTA LSH** : Cross-Polytope LSH avec Multi-Probe (hachage géométrique sans apprentissage)
3. **HeCBR** : Deep Hashing supervisé (réseau MLP 512→256→64 bits, recherche par distance de Hamming)

---

## Accès SSH

- **Machine locale** : Ubuntu Linux
- **Clé SSH** : `~/.ssh/lightning_rsa` (déjà configurée)
- **Fichier `~/.ssh/config`** contient déjà :
  ```
  Host ssh.lightning.ai
    IdentityFile ~/.ssh/lightning_rsa
    IdentitiesOnly yes
    StrictHostKeyChecking no
  ```
- **Utilisateur SSH probable** : `s_01m1141xqcrf6m2zzpya9xrhne`
- **Hôte** : `ssh.lightning.ai`
- **Commande SSH de base** : `ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "commande"`
- **Python sur le serveur** : `/home/zeus/miniconda3/envs/cloudspace/bin/python`
- **Répertoire de travail sur le serveur** : `/home/zeus/`

Si le studio est arrêté, le démarrer avec :
```bash
lightning studio start --teamspace "financial-llm-training-project" --name "get-camelyon-resnet50"
```
Puis attendre 60-90 secondes avant de retenter la connexion SSH.

---

## Code source à déployer

- **Source locale** : `/home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/`
- **Destination sur le serveur** : `/home/zeus/hecbr_sota_camelyon/`

Structure du projet :
```
hecbr_sota_camelyon/
├── config/         protocol.py, paths.py, seeds.py
├── data/           loader.py, splitter.py, preprocessing.py
├── indexes/        linear_scan.py, cross_polytope.py, hecbr_index.py
├── models/         hecbr_net.py, train_hecbr.py
├── benchmark/      timing.py, metrics.py
├── visualization/  theme.py, plots.py
├── scripts/        train_hecbr.py, run_comparative_benchmark.py
└── tests/          test_linear_scan.py, test_cross_polytope.py, test_hecbr.py
```

**Cache Camelyon16 déjà sur le serveur** (4,88M patches, 9,4 Go) — **NE PAS retransférer** :
```
/home/zeus/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl
```
Le `config/paths.py` du projet pointe déjà vers ce cache.

**NE JAMAIS toucher** au dossier `/home/zeus/lsh_camelyon/` (expérience précédente archivée).

---

## Étapes à exécuter dans l'ordre strict

### Étape 0 — Vérifications préliminaires
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "echo OK && hostname"
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "ls -lh /home/zeus/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl"
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "/home/zeus/miniconda3/envs/cloudspace/bin/python -c 'import torch, sklearn, numpy, pandas; print(torch.__version__, sklearn.__version__)'"
```

Si le cache n'est pas à ce chemin, le chercher avec :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "find /home/zeus -name 'camelyon16_slides_cache.pkl' 2>/dev/null"
```
Et mettre à jour la variable `CACHE_FILE` dans `config/paths.py` localement.

### Étape 1 — Transfert du code source
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "mkdir -p /home/zeus/hecbr_sota_camelyon/logs \
             /home/zeus/hecbr_sota_camelyon/results/raw \
             /home/zeus/hecbr_sota_camelyon/results/figures \
             /home/zeus/hecbr_sota_camelyon/models/checkpoints"

rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='results/' --exclude='models/checkpoints/' \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/ \
  s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/
```

Vérification :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai "ls /home/zeus/hecbr_sota_camelyon/"
```

### Étape 2 — Tests unitaires sur le serveur
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "cd /home/zeus/hecbr_sota_camelyon && \
   /home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest \
   tests/test_linear_scan.py tests/test_cross_polytope.py tests/test_hecbr.py -v"
```

**Résultat attendu : 5 tests PASSED.** Si un test échoue, corriger localement et re-transférer uniquement le fichier corrigé avant de continuer.

### Étape 3 — Entraînement du modèle HeCBR (~15-25 minutes)
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "cd /home/zeus/hecbr_sota_camelyon && \
   nohup /home/zeus/miniconda3/envs/cloudspace/bin/python scripts/train_hecbr.py \
     --bits 64 --epochs 15 --batch-size 1024 --lr 0.001 --max-train-samples 300000 \
   > /home/zeus/hecbr_sota_camelyon/logs/train_hecbr.log 2>&1 & echo PID=\$!"
```

Monitoring :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "tail -20 /home/zeus/hecbr_sota_camelyon/logs/train_hecbr.log"
```

Signe de succès : log se termine par `Modèle sauvegardé dans` + fichier `.pt` présent :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "ls -lh /home/zeus/hecbr_sota_camelyon/models/checkpoints/hecbr_camelyon16_64bit.pt"
```

### Étape 4 — Benchmark comparatif complet (~2-4 heures)
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "cd /home/zeus/hecbr_sota_camelyon && \
   nohup /home/zeus/miniconda3/envs/cloudspace/bin/python \
     scripts/run_comparative_benchmark.py --k 10 \
   > /home/zeus/hecbr_sota_camelyon/logs/benchmark.log 2>&1 & echo PID=\$!"
```

Monitoring (toutes les 10-15 minutes) :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "tail -30 /home/zeus/hecbr_sota_camelyon/logs/benchmark.log"
```

Étapes visibles dans le log (dans l'ordre) :
1. `=== Étape 1 : Chargement des données Camelyon16 ===`
2. `=== 1. Baseline Linéaire BLAS (Exacte) ===`
3. `=== 2. SOTA LSH : Cross-Polytope avec Multi-Probe ===`
4. `=== 3. HeCBR : Deep Hashing 64-bit (Distance de Hamming) ===`

Signe de succès : `=== Benchmark comparatif complet terminé en XXX s ===`

Vérification des fichiers produits :
```bash
ssh s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai \
  "ls -lh /home/zeus/hecbr_sota_camelyon/results/raw/ && \
   ls -lh /home/zeus/hecbr_sota_camelyon/results/figures/"
```

### Étape 5 — Rapatriement local des résultats
```bash
rsync -av s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/results/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/

rsync -av s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/logs/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/logs/

rsync -av s_01m1141xqcrf6m2zzpya9xrhne@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/models/checkpoints/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/models/checkpoints/
```

Vérification locale :
```bash
ls -lh /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/
cat /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/benchmark_timing.csv
cat /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/benchmark_quality.csv
```

---

## Gestion des erreurs

| Erreur | Solution |
|--------|----------|
| `Permission denied (publickey)` | Le studio est arrêté. Lancer : `lightning studio start --teamspace "financial-llm-training-project" --name "get-camelyon-resnet50"` puis attendre 90s |
| `ModuleNotFoundError: torch` | `ssh ... "/home/zeus/miniconda3/envs/cloudspace/bin/pip install torch --quiet"` |
| Cache pickle introuvable | `ssh ... "find /home/zeus -name 'camelyon16_slides_cache.pkl'"` puis mettre à jour `config/paths.py` |
| OOM pendant le benchmark | `ssh ... "free -h"` pour vérifier la RAM. Si < 10 Go libres, arrêter les autres processus |
| Studio éteint pendant le benchmark | Redémarrer le studio, puis `ssh ... "ps aux | grep python"` pour vérifier si le `nohup` tourne encore |

---

## Rapport final attendu

À la fin de l'Étape 5, fournis un rapport structuré contenant :
1. Tableau des **latences médianes** par méthode et seed (en ms)
2. Tableau des **métriques qualité** par méthode et seed (Recall@10, Macro-F1)
3. **Taille du fichier modèle HeCBR** (en Mo)
4. **Durées** : entraînement HeCBR (min) + benchmark total (min)
5. Toute **anomalie ou erreur** rencontrée avec son message complet
