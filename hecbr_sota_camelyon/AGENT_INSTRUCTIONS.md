# Instructions opérationnelles — Agent de déploiement Lightning AI
# Expérience : HeCBR + SOTA LSH (Cross-Polytope) vs Baseline sur Camelyon16

> **IMPORTANT — Avant de faire quoi que ce soit :**
> Lis l'intégralité de ce fichier. Ensuite, dans ta première réponse, décris **exactement ce que tu comptes faire, étape par étape, avec les commandes précises**. Ne lance aucune commande tant que l'agent parent ne t'a pas confirmé « OK, go ».

---

## 1. Contexte du projet de recherche

Tu es en train de poursuivre une expérience de recherche sur la **comparaison des méthodes de recherche de plus proches voisins à très grande échelle** ($3,77 \times 10^6$ patches histopathologiques, 512 dimensions) sur la base médicale **Camelyon16**.

La précédente expérience (SimHash pur, archivée dans `/home/zeus/lsh_camelyon/`) a établi :
1. À l'échelle $10^6+$, le LSH naïf (SimHash) est **1,42x plus rapide** que le balayage linéaire.
2. Mais SimHash s'effondre en qualité : **Recall@10 ≈ 1%** (au lieu de 100%), à cause de la géométrie de la dimension $d=512$.

La **nouvelle expérience** que tu dois exécuter introduit deux méthodes de pointe :
- **SOTA LSH (Cross-Polytope avec Multi-Probe)** : Hachage géométrique optimal sans apprentissage.
- **HeCBR (Deep Hashing supervisé)** : Réseau MLP 512→256→64 bits, recherche en distance de Hamming.

---

## 2. Accès SSH à Lightning AI

**Clé SSH** : `~/.ssh/lightning_rsa` (déjà configurée dans `~/.ssh/config` avec l'hôte `ssh.lightning.ai`).

**Utilisateur** : À récupérer depuis le nom de ton studio en cours. Utilise la commande suivante pour trouver le bon utilisateur :
```bash
lightning ssh --list 2>/dev/null || cat ~/.ssh/config
```

La forme du nom d'utilisateur est `s_XXXXXXXXXXXXXXXXXXXXXXXXXX`.

**Si le studio est arrêté**, tu dois d'abord le démarrer avec la commande CLI Lightning :
```bash
lightning studio start --teamspace "financial-llm-training-project" --name "get-camelyon-resnet50"
```
Puis vérifier qu'il est accessible :
```bash
ssh s_XXXXXXXXXXXXXXXXX@ssh.lightning.ai "echo OK"
```

**Interpréteur Python sur le serveur distant** :
```bash
PYTHON="/home/zeus/miniconda3/envs/cloudspace/bin/python"
```
**Répertoire de travail sur le serveur** : `/home/zeus/`

---

## 3. Code source local à déployer

**Répertoire source local (ta machine)** :
```
/home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/
```

**Destination sur le serveur distant** :
```
/home/zeus/hecbr_sota_camelyon/
```

**Cache Camelyon16 déjà sur le serveur** (4,88M patches, 9,4 Go, NE PAS retransférer) :
```
/home/zeus/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl
```
Le fichier `config/paths.py` du code source pointe déjà vers ce cache en premier choix.

---

## 4. Étapes à exécuter — dans cet ordre strict

### Étape 0 : Vérification préliminaire (OBLIGATOIRE avant tout déploiement)

**Avant de transférer quoi que ce soit**, vérifie sur le serveur distant :

```bash
# 1. Vérifier que le studio est accessible
ssh s_XXXX@ssh.lightning.ai "echo '=== Studio OK ===' && hostname"

# 2. Vérifier que le cache Camelyon16 existe
ssh s_XXXX@ssh.lightning.ai "ls -lh /home/zeus/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl"

# 3. Vérifier que PyTorch est disponible
ssh s_XXXX@ssh.lightning.ai "/home/zeus/miniconda3/envs/cloudspace/bin/python -c 'import torch; print(\"Torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'"

# 4. Vérifier sklearn, numpy, pandas
ssh s_XXXX@ssh.lightning.ai "/home/zeus/miniconda3/envs/cloudspace/bin/python -c 'import sklearn, numpy, pandas; print(\"sklearn:\", sklearn.__version__, \"numpy:\", numpy.__version__)'"
```

Si le cache n'existe pas au chemin indiqué, cherche-le avec :
```bash
ssh s_XXXX@ssh.lightning.ai "find /home/zeus -name 'camelyon16_slides_cache.pkl' 2>/dev/null"
```
Et mets à jour `config/paths.py` localement avant de déployer (cherche la variable `CACHE_FILE`).

---

### Étape 1 : Transfert du code source

Utilise `rsync` (pas `scp`) pour n'envoyer que les différences :

```bash
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='results/' --exclude='models/checkpoints/' \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/ \
  s_XXXX@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/
```

Vérifie que le transfert a réussi :
```bash
ssh s_XXXX@ssh.lightning.ai "ls /home/zeus/hecbr_sota_camelyon/"
```

Résultat attendu : les dossiers `config/`, `data/`, `indexes/`, `models/`, `benchmark/`, `visualization/`, `scripts/`, `tests/` sont présents.

---

### Étape 2 : Validation des tests unitaires sur le serveur

```bash
ssh s_XXXX@ssh.lightning.ai "cd /home/zeus/hecbr_sota_camelyon && \
  /home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest tests/test_linear_scan.py tests/test_cross_polytope.py tests/test_hecbr.py -v"
```

**Résultat attendu : 5 tests verts (PASSED)**. Si un test échoue :
- Lis le message d'erreur.
- Corrige le bug dans le fichier concerné **localement**.
- Re-transfère uniquement le fichier corrigé avec `rsync`.
- Relance les tests.
- Ne passe pas à l'Étape 3 tant que tous les tests ne passent pas.

---

### Étape 3 : Entraînement du modèle HeCBR (Deep Hashing)

Cette étape entraîne un petit réseau MLP (512→256→64 bits) sur un sous-échantillon représentatif de 300 000 patches de la base Camelyon16 (extrait automatiquement depuis le cache).

```bash
ssh s_XXXX@ssh.lightning.ai "cd /home/zeus/hecbr_sota_camelyon && \
  nohup /home/zeus/miniconda3/envs/cloudspace/bin/python scripts/train_hecbr.py \
    --bits 64 --epochs 15 --batch-size 1024 --lr 0.001 \
    --max-train-samples 300000 \
  > /home/zeus/hecbr_sota_camelyon/logs/train_hecbr.log 2>&1 &
  echo 'PID:' $!"
```

**Durée estimée** : 15 à 25 minutes (CPU uniquement), moins si GPU disponible.

**Monitoring de la progression** :
```bash
# Vérifier que le processus tourne
ssh s_XXXX@ssh.lightning.ai "ps aux | grep train_hecbr | grep -v grep"

# Afficher les dernières lignes du log
ssh s_XXXX@ssh.lightning.ai "tail -20 /home/zeus/hecbr_sota_camelyon/logs/train_hecbr.log"
```

**Signe de succès** : le log se termine par une ligne contenant `Modèle sauvegardé dans` et le fichier suivant doit exister :
```bash
ssh s_XXXX@ssh.lightning.ai "ls -lh /home/zeus/hecbr_sota_camelyon/models/checkpoints/hecbr_camelyon16_64bit.pt"
```

---

### Étape 4 : Benchmark comparatif complet (Baseline vs SOTA LSH vs HeCBR)

C'est l'étape principale de l'expérience. Elle prend **2 à 4 heures**.

```bash
ssh s_XXXX@ssh.lightning.ai "cd /home/zeus/hecbr_sota_camelyon && \
  mkdir -p logs results/raw results/figures && \
  nohup /home/zeus/miniconda3/envs/cloudspace/bin/python \
    scripts/run_comparative_benchmark.py --k 10 \
  > /home/zeus/hecbr_sota_camelyon/logs/benchmark.log 2>&1 &
  echo 'PID:' $!"
```

**Monitoring** :
```bash
# Afficher les dernières lignes du log toutes les 5-10 minutes
ssh s_XXXX@ssh.lightning.ai "tail -30 /home/zeus/hecbr_sota_camelyon/logs/benchmark.log"
```

**Étapes visible dans le log** (dans l'ordre) :
1. `=== Étape 1 : Chargement des données Camelyon16 ===` → charge le cache pickle
2. `=== 1. Baseline Linéaire BLAS (Exacte) ===` → benchmark de référence
3. `=== 2. SOTA LSH : Cross-Polytope avec Multi-Probe ===` → 5 seeds
4. `=== 3. HeCBR : Deep Hashing 64-bit (Distance de Hamming) ===` → 5 seeds

**Signe de succès** : le log se termine par :
```
=== Benchmark comparatif complet terminé en XXX s ===
```

Et les fichiers suivants doivent exister :
```bash
ssh s_XXXX@ssh.lightning.ai "ls -lh /home/zeus/hecbr_sota_camelyon/results/raw/ && ls -lh /home/zeus/hecbr_sota_camelyon/results/figures/"
```

Résultats attendus dans `results/raw/` :
- `benchmark_timing.csv` : latences médianes, moyennes, p25, p75, p95, p99 par méthode et seed.
- `benchmark_quality.csv` : Recall@10, Macro-F1 par méthode et seed.
- `ground_truth_test.npz` : indices des vrais voisins exacts (vérité terrain).

Figures attendues dans `results/figures/` :
- `fig1_latency_comparison.png`
- `fig2_speed_quality.png`
- `fig3_memory_footprint.png`

---

### Étape 5 : Rapatriement des résultats en local

```bash
rsync -av s_XXXX@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/results/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/

rsync -av s_XXXX@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/logs/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/logs/

rsync -av s_XXXX@ssh.lightning.ai:/home/zeus/hecbr_sota_camelyon/models/checkpoints/ \
  /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/models/checkpoints/
```

**Vérifie localement que les fichiers sont arrivés** :
```bash
ls -lh /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/
cat /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/benchmark_timing.csv | head -20
cat /home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/results/raw/benchmark_quality.csv | head -20
```

---

### Étape 6 : Rapport de résultats à l'agent parent

Une fois tous les fichiers rapatriés localement, envoie un **message structuré** à l'agent parent (moi) contenant **exactement** :

1. **Tableau des latences médianes** (au format markdown) : méthode × seed × latence médiane en ms.
2. **Tableau des métriques qualité** : méthode × seed × Recall@10 × Macro-F1.
3. **Taille du fichier modèle HeCBR** (en Mo).
4. **Durée totale** de l'entraînement (en minutes) et du benchmark (en minutes).
5. **Toute anomalie ou erreur observée** pendant l'exécution, avec le message d'erreur complet.

---

## 5. Gestion des erreurs fréquentes

### Erreur : `ModuleNotFoundError` sur le serveur
```bash
# Vérifier les modules disponibles
ssh s_XXXX@ssh.lightning.ai "/home/zeus/miniconda3/envs/cloudspace/bin/pip list | grep -E 'torch|sklearn|numpy|pandas'"

# Si PyTorch manque (très peu probable sur Lightning AI) :
ssh s_XXXX@ssh.lightning.ai "/home/zeus/miniconda3/envs/cloudspace/bin/pip install torch --quiet"
```

### Erreur : Cache Camelyon16 introuvable au chemin configuré
1. Localise le cache : `find /home/zeus -name 'camelyon16_slides_cache.pkl'`
2. Mets à jour la variable `CACHE_FILE` dans `config/paths.py` **localement** avec le bon chemin.
3. Re-transfère uniquement `config/paths.py` : `rsync ... hecbr_sota_camelyon/config/paths.py ...`

### Erreur : `CUDA out of memory` pendant l'entraînement
```bash
# Forcer le CPU uniquement
ssh s_XXXX@ssh.lightning.ai "cd /home/zeus/hecbr_sota_camelyon && \
  CUDA_VISIBLE_DEVICES='' /home/zeus/miniconda3/envs/cloudspace/bin/python \
  scripts/train_hecbr.py --bits 64 --epochs 15 --batch-size 512 \
  > logs/train_hecbr.log 2>&1 &"
```

### Erreur : OOM (Out of Memory) pendant le benchmark
Le script charge 3,77M patches float32 → ~7,5 Go de RAM. Si le serveur manque de RAM :
1. Vérifie la RAM disponible : `ssh ... "free -h"`
2. Si < 10 Go libres, arrête les autres processus sur le serveur.

### Le studio Lightning AI s'est arrêté pendant l'exécution
- Rallume-le : `lightning studio start --teamspace "financial-llm-training-project" --name "get-camelyon-resnet50"`
- Attends 60-90 secondes puis reconnecte-toi.
- Vérifie si le processus `nohup` tourne encore : `ssh ... "ps aux | grep python | grep -v grep"`
- Si le processus est mort, relance l'étape interrompue depuis le début (le cache et la vérité terrain sont déjà sauvegardés sur disque s'ils ont été calculés).

---

## 6. Chemin de travail de l'agent parent

- **Identifiant de la conversation parent** : `ba38d35c-8898-4af0-ad30-3d8b12251ae5`
- **Répertoire local du projet** : `/home/customizer/Downloads/Research/Hashing/code/hecbr_sota_camelyon/`
- **Répertoire archivé de l'expérience précédente (SimHash, NE PAS TOUCHER)** : `/home/customizer/Downloads/Research/Hashing/code/lsh_camelyon/`
- **Fichier de log SSH utilisé lors des précédentes expériences** : `~/.ssh/lightning_rsa`

---

## 7. Règles absolues

1. **Ne touche JAMAIS** au dossier `/home/zeus/lsh_camelyon/` sur le serveur (expérience archivée).
2. **Ne touche JAMAIS** au dossier local `/home/customizer/Downloads/Research/Hashing/code/lsh_camelyon/`.
3. **Ne lance aucune commande** avant d'avoir décrit ton plan complet à l'agent parent.
4. **Si une étape échoue**, ne passe pas à la suivante. Diagnostique l'erreur, propose une correction à l'agent parent, attends sa validation.
5. **Toutes les sorties importantes** doivent être redirigées vers un fichier de log dans `logs/` avec `nohup ... > logs/xxx.log 2>&1 &`.
6. **Ne modifie aucun fichier de l'expérience précédente SimHash**, ni les CSV de résultats dans `lsh_camelyon/results/`.
