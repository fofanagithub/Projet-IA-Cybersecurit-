"""04_leave_attack_out.py — Simulation de zero-day À L'INTÉRIEUR du train.

Pour chaque type d'attaque A du train (n ≥ 10) : on entraîne sans aucune
connexion de type A (et sans 20 % des connexions normales, mises de côté),
puis on mesure le taux de détection de A et le FPR sur les normales mises de côté.
Question de recherche : ce protocole, qui n'utilise que le train, prédit-il
la capacité d'un modèle à détecter les attaques réellement inédites du test ?
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, make_preprocessor, FEATURES
from models import supervised

ROOT = Path(__file__).resolve().parents[1]
tr = load("train")
counts = tr.loc[tr.is_attack == 1, "attack"].value_counts()
TYPES = counts[counts >= 10].index.tolist()
MODELS = ["LogReg", "DecisionTree", "RandomForest", "LightGBM", "MLP"]

norm_idx = np.where(tr.is_attack.values == 0)[0]
_, norm_hold = train_test_split(norm_idx, test_size=0.2, random_state=0)
hold = np.zeros(len(tr), bool); hold[norm_hold] = True

out = ROOT / "results" / "leave_attack_out.json"
R = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
for A in TYPES:
    is_A = (tr.attack == A).values
    fit_mask = ~is_A & ~hold
    pre = make_preprocessor().fit(tr.loc[fit_mask, FEATURES])     # refait sans A : aucune fuite
    Xf = pre.transform(tr.loc[fit_mask, FEATURES]); yf = tr.is_attack.values[fit_mask]
    XA = pre.transform(tr.loc[is_A, FEATURES]); XN = pre.transform(tr.loc[hold, FEATURES])
    for m in MODELS:
        key = f"{A}|{m}"
        if key in R:
            continue
        t = time.time()
        clf = supervised(m, 0).fit(Xf, yf)
        R[key] = {"attack": A, "category": tr.loc[is_A, "category"].iloc[0], "n": int(is_A.sum()), "model": m,
                  "DR": float((clf.predict_proba(XA)[:, 1] >= .5).mean()),
                  "FPR": float((clf.predict_proba(XN)[:, 1] >= .5).mean())}
        print(f"{A:15s} {m:13s} DR={R[key]['DR']:.3f} FPR={R[key]['FPR']:.4f} ({time.time()-t:.0f}s)", flush=True)
        out.write_text(json.dumps(R, indent=1), encoding="utf-8")
print("TERMINÉ")

# ---- Synthèse : moyenne par modèle et corrélation avec la détection des inédits du test
import pandas as pd
from scipy.stats import spearmanr
d = pd.DataFrame(R.values())
pv = d.pivot(index=["attack", "category", "n"], columns="model", values="DR")[MODELS]
an = json.loads((ROOT / "results" / "analysis.json").read_text(encoding="utf-8"))["recall_seen_novel"]
novel = pd.Series({m: an[m]["novel"][0] for m in MODELS})
tb = pd.read_csv(ROOT / "results" / "table_binary.csv")
cv = tb[tb["Évaluation"].str.startswith("CV")].set_index("Modèle")["_F1"][MODELS]
(ROOT / "results" / "lao_summary.json").write_text(json.dumps({
    "macro_DR": pv.mean().to_dict(), "FPR": d.groupby("model").FPR.mean().to_dict(),
    "spearman_lao_vs_test_novel": spearmanr(pv.mean()[MODELS], novel)[0],
    "spearman_cvF1_vs_test_novel": spearmanr(cv, novel)[0],
    "table": pv.reset_index().to_dict(orient="records")}, indent=1, default=float), encoding="utf-8")
print(pv.round(2).to_string())
