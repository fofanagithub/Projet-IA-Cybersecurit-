"""05_explain_ablation.py — Explicabilité (SHAP) et ablation par groupes de variables."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, make_preprocessor, novel_mask, FEATURES, FEATURE_GROUPS, CATEGORICAL
from models import supervised
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

tr, te = load("train"), load("test")
y, ytr = te.is_attack.values, tr.is_attack.values
NOVEL = novel_mask(tr, te).values
SEEN = (y == 1) & ~NOVEL
R = {}

# ---------------- SHAP sur LightGBM binaire
pre = make_preprocessor().fit(tr[FEATURES])
Xtr, Xte = pre.transform(tr[FEATURES]), pre.transform(te[FEATURES])
names = pre.get_feature_names_out()
orig = np.array([next((c for c in CATEGORICAL if n.startswith(c + "_")), n) for n in names])
clf = supervised("LightGBM", 0).fit(Xtr, ytr)
rng = np.random.default_rng(0)
sub = rng.choice(len(te), 4000, replace=False)
sv = shap.TreeExplainer(clf).shap_values(Xte[sub])
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=names).groupby(orig).sum().sort_values(ascending=False)
R["shap_top15"] = imp.head(15).round(4).to_dict()
grp_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
R["shap_by_group_share"] = (imp.groupby(imp.index.map(grp_of)).sum() / imp.sum()).round(4).to_dict()

# SHAP sur attaques inédites manquées vs détectées : qu'est-ce qui « rassure » le modèle ?
p = clf.predict_proba(Xte)[:, 1]
missed = np.where(NOVEL & (p < .5))[0]
mi = rng.choice(missed, min(1500, len(missed)), replace=False)
svm = shap.TreeExplainer(clf).shap_values(Xte[mi]); svm = svm[1] if isinstance(svm, list) else svm
push_normal = pd.Series(svm.mean(0), index=names).groupby(orig).sum().sort_values()
R["shap_missed_novel_most_negative"] = push_normal.head(10).round(4).to_dict()
R["missed_novel_types"] = te.iloc[missed]["attack"].value_counts().head(8).to_dict()

fig, ax = plt.subplots(figsize=(6.5, 4.6))
top = imp.head(15)[::-1]
cols = [C[list(FEATURE_GROUPS).index(grp_of[f])] for f in top.index]
ax.barh(range(len(top)), top.values, color=cols, height=.7)
ax.set_yticks(range(len(top)), top.index, fontsize=8.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("|SHAP| moyen (log-odds), modalités one-hot regroupées par variable")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C[i], label=g) for i, g in enumerate(FEATURE_GROUPS)], fontsize=7.5, loc="lower right")
ax.set_title("LightGBM : aucune variable de contenu parmi les 15 plus influentes")
fig.savefig(FIG / "fig8_shap.png"); plt.close(fig)

# ---------------- Ablation : retirer un groupe de variables à la fois
abl = {}
for g, fs in [("aucune (toutes variables)", [])] + list(FEATURE_GROUPS.items()):
    keep = [f for f in FEATURES if f not in fs]
    num =[f for f in keep if f not in CATEGORICAL]; cat = [f for f in keep if f in CATEGORICAL]
    ct = ColumnTransformer([("n", StandardScaler(), num)] +
                           ([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)] if cat else []))
    A, Bt = ct.fit_transform(tr[keep]), ct.transform(te[keep])
    res = []
    for s in [0, 1, 2]:
        pp = supervised("LightGBM", s).fit(A, ytr).predict_proba(Bt)[:, 1]
        res.append({"F1": f1_score(y, pp >= .5), "AUC": roc_auc_score(y, pp),
                    "DR_seen": (pp[SEEN] >= .5).mean(), "DR_novel": (pp[NOVEL] >= .5).mean(),
                    "FPR": (pp[y == 0] >= .5).mean()})
    abl[g] = pd.DataFrame(res).mean().round(4).to_dict()
    print(g, abl[g], flush=True)
R["ablation_lightgbm"] = abl
(ROOT / "results" / "explain_ablation.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(R, indent=1, ensure_ascii=False, default=float))
