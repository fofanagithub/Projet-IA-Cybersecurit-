"""01_eda.py — Diagnostic du jeu de données avant toute modélisation."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, novel_mask, FEATURES, CATEGORICAL, CLASSES

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
R = {}

tr, te, te21 = load("train"), load("test"), load("test21")
for name, d in [("train", tr), ("test", te), ("test21", te21)]:
    R[f"n_{name}"] = len(d)
    R[f"cat_{name}"] = d["category"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
    R[f"n_attack_types_{name}"] = int(d.loc[d.is_attack == 1, "attack"].nunique())

# --- Attaques inédites (présentes au test, absentes du train)
nm = novel_mask(tr, te)
R["novel_types"] = sorted(te.loc[nm, "attack"].unique().tolist())
R["novel_count_test"] = int(nm.sum())
R["novel_share_of_test_attacks"] = float(nm.sum() / te.is_attack.sum())
R["novel_by_category"] = te.loc[nm, "category"].value_counts().to_dict()
R["novel_share_by_category"] = (te.loc[nm, "category"].value_counts()
                                / te.loc[te.is_attack == 1, "category"].value_counts()).round(4).to_dict()
nm21 = novel_mask(tr, te21)
R["novel_share_of_test21_attacks"] = float(nm21.sum() / te21.is_attack.sum())

# --- Qualité : doublons, conflits d'étiquettes, variables constantes
R["dup_rows_train"] = int(tr.duplicated(subset=FEATURES + ["attack"]).sum())
g = tr.groupby(FEATURES, dropna=False)["category"].nunique()
R["conflicting_feature_vectors_train"] = int((g > 1).sum())
R["constant_features_train"] = [c for c in FEATURES if tr[c].nunique() == 1]
R["unseen_services_in_test"] = sorted(set(te.service) - set(tr.service))
# vecteurs identiques train/test
key = lambda d: pd.util.hash_pandas_object(d[FEATURES], index=False)
R["test_rows_identical_to_a_train_row"] = int(key(te).isin(set(key(tr))).sum())

# --- Distribution des difficultés
R["difficulty_mean"] = {"train": float(tr.difficulty.mean()), "test": float(te.difficulty.mean())}

# --- Validation adversariale : un classifieur sait-il distinguer train et test ?
# AUC ≈ 0.5 → même distribution ; AUC ≫ 0.5 → décalage de distribution (covariate shift)
both = pd.concat([tr[FEATURES].assign(_is_test=0), te[FEATURES].assign(_is_test=1)], ignore_index=True)
X = both[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("category")
y = both["_is_test"].values
clf = HistGradientBoostingClassifier(max_iter=200, categorical_features="from_dtype", random_state=0)
p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
R["adversarial_validation_auc_all"] = float(roc_auc_score(y, p))
# restreint au trafic normal : le décalage existe-t-il hors attaques ?
lab = pd.concat([tr["category"], te["category"]], ignore_index=True)
m = (lab == "Normal").values
p2 = cross_val_predict(clf, X[m], y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
R["adversarial_validation_auc_normal_only"] = float(roc_auc_score(y[m], p2))

# Attaques par type (pour le rapport)
R["attack_counts_train"] = tr.loc[tr.is_attack == 1, "attack"].value_counts().to_dict()
R["attack_counts_test"] = te.loc[te.is_attack == 1, "attack"].value_counts().to_dict()

(OUT / "eda.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in R.items() if not k.startswith("attack_counts")}, indent=1, ensure_ascii=False))
