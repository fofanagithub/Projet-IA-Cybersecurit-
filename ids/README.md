# Détection d'intrusions par apprentissage automatique sur NSL-KDD
### Une évaluation sous décalage de distribution et face aux attaques inédites

Projet reproductible accompagnant le rapport. Chaque chiffre du rapport est produit par
les scripts ci-dessous ; aucun n'est saisi à la main.

## Structure

```
ids/
├── data/raw/                  KDDTrain+, KDDTest+, KDDTest-21 (téléchargés, cf. ci-dessous)
├── src/
│   ├── nslkdd.py              chargement, taxonomie DARPA, prétraitement sans fuite
│   ├── models.py              5 modèles supervisés + 2 détecteurs d'anomalies
│   ├── 01_eda.py              diagnostic du jeu de données (décalage, attaques inédites…)
│   ├── 02_experiments.py      entraînement (5 graines), prédictions sauvegardées
│   ├── 03_analysis.py         métriques, IC bootstrap, McNemar-Holm, calibration, figures
│   ├── 04_leave_attack_out.py simulation de zero-day à l'intérieur du train
│   ├── 05_explain_ablation.py SHAP + ablation par groupes de variables
│   ├── 06_host_ablation_per_type.py  effet par type du retrait des variables host-based
│   └── 99_verify_report.py    recalcule les chiffres cités dans le rapport (avec tolérances)
├── requirements.txt           versions exactes des dépendances
└── results/                   JSON/CSV de résultats + figures/
```

## Reproduire

```bash
pip install -r requirements.txt
mkdir -p data/raw && cd data/raw
for f in KDDTrain+.txt KDDTest+.txt KDDTest-21.txt; do
  curl -sSfLO "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/$f"; done
cd ../..
python src/01_eda.py && python src/02_experiments.py && python src/03_analysis.py
python src/04_leave_attack_out.py && python src/05_explain_ablation.py
python src/06_host_ablation_per_type.py && python src/99_verify_report.py   # contrôle des chiffres du rapport
```

Empreintes SHA-256 des fichiers utilisés (contrôle d'intégrité) :

| Fichier | Lignes | SHA-256 |
|---|---|---|
| KDDTrain+.txt | 125 973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| KDDTest+.txt | 22 544 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |
| KDDTest-21.txt | 11 850 | `746993ac9e25868827cacf09eab450050a2a1056e1ce48a1ad39f5dc801d531d` |

Matériel de référence : 2 vCPU, 7 Go RAM, aucun GPU — durée totale ≈ 1 h 30
(≈ 1 h 10 sur une machine Windows 11 à 8 cœurs, Python 3.12).

**Windows** : les scripts fonctionnent tels quels. Toutes les lectures/écritures de fichiers
sont explicitement en UTF-8 et la sortie console est reconfigurée en UTF-8, donc l'encodage
par défaut de Windows (cp1252) ne provoque plus d'`UnicodeEncodeError` ; aucune variable
`PYTHONUTF8` n'est nécessaire. Sous PowerShell, la boucle `curl` ci-dessus se lance depuis
Git Bash, ou se remplace par `Invoke-WebRequest`.

## Reproductibilité numérique

Une reproduction complète sur une autre machine que celle d'origine donne :

| Modèles | Écart observé |
|---|---|
| DecisionTree, RandomForest, LightGBM, IsolationForest, auto-encodeur | aucun (identiques au bit près) |
| LogReg (lbfgs), MLP (Adam) | quelques dixièmes de point (ex. F1 MLP sur KDDTest+ : 79,68 → 79,93) |

Ces écarts viennent de l'algèbre linéaire sous-jacente (bibliothèque BLAS, nombre de threads,
système d'exploitation), pas d'une graine mal fixée. Ils ne changent aucune conclusion, mais
ils déplacent les chiffres qui dépendent de ces deux modèles. `99_verify_report.py` en tient
compte :

- tolérance = arrondi affiché pour les chiffres issus des modèles à arbres ;
- `TOL_NUM` = 0,5 point pour les min/max calculés sur les 5 modèles et pour le F1 macro du MLP ;
- `TOL_RANK` = 0,1 pour les corrélations de Spearman. Avec 5 modèles, ρ ne prend que des
  multiples de 0,1 et une seule inversion de rang entre deux modèles quasi ex aequo le décale
  de 0,1 (observé : 0,20 → 0,10, le MLP passant devant l'arbre d'un centième de point).
  Ces corrélations sont donc **indicatives** et non significatives.

Pour supprimer l'essentiel de la variabilité sur une même machine, fixer le nombre de threads
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) avant de lancer les scripts.

## Garde-fous méthodologiques

1. **Aucune fuite** : encodage et normalisation appris sur le train seul (`sklearn.Pipeline`) ;
   la colonne `difficulty` n'est jamais une variable explicative.
2. **Aucune décision prise sur le test** : seuils et choix de modèle viennent de la validation
   croisée hors-pli ou d'une validation interne au train. Cela vaut aussi pour le test de
   McNemar : sa référence est le modèle retenu par la CV (`best_by_cv`, LightGBM), et non le
   meilleur sur le test — ce dernier choix serait à la fois une décision prise sur le test et
   instable (arbre et MLP sont ex aequo à 0,01 point près).
3. **Incertitude quantifiée** : 5 graines, IC bootstrap à 95 %, test de McNemar exact avec
   correction de Holm (résultats dans `analysis.json`, clé `mcnemar_vs_LightGBM`).
4. **Évaluation désagrégée** : attaques vues vs inédites, par type, KDDTest+ et KDDTest-21.

## Passer à CIC-IDS2017

Le pipeline est générique : il suffit de réécrire `load()` et la liste des variables
dans `nslkdd.py`. Avant de le faire, lire Engelen et al. (2021) et Lanvin et al. (2022),
qui documentent des erreurs d'étiquetage et de génération de flux dans la version
originale de CIC-IDS2017 ; utiliser la version corrigée, et découper train/test **par
jour de capture** et non aléatoirement, sous peine de reproduire l'illusion décrite
à la section 5.2 du rapport.
