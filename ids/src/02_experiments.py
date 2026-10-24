"""02_experiments.py — Entraînement de tous les modèles, prédictions sauvegardées.

Sorties (results/preds/) :
  bin_<model>_s<seed>.npy        P(attaque) sur KDDTest+
  mc_<model>[_bal]_s<seed>.npy   probabilités 5 classes sur KDDTest+
  oof_bin_<model>.npy            probas hors-pli (CV 5 plis) sur KDDTrain+  (seed 0)
  ad_<det>_s<seed>_{test,valnorm}.npy  scores d'anomalie (test, normal de validation)
KDDTest-21 = KDDTest+ restreint à difficulty < 21 (vérifié) → obtenu par masque.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, make_preprocessor, CLASSES, FEATURES
from models import SUPERVISED, supervised, AutoEncoderDetector, IForestDetector

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "preds"
P.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
timing = {}

tr, te = load("train"), load("test")
pre = make_preprocessor().fit(tr[FEATURES])           # appris sur le TRAIN uniquement
Xtr = pre.transform(tr[FEATURES]).astype(np.float32)
Xte = pre.transform(te[FEATURES]).astype(np.float32)
ybin = tr["is_attack"].values
ymc = tr["category"].map({c: i for i, c in enumerate(CLASSES)}).values
print("dim. après encodage :", Xtr.shape[1], flush=True)
json.dump(pre.get_feature_names_out().tolist(), open(ROOT / "results" / "feature_names.json", "w", encoding="utf-8"))


def run(tag, fn):
    out = P / f"{tag}.npy"
    if out.exists():
        return
    t = time.time()
    np.save(out, fn())
    timing[tag] = round(time.time() - t, 1)
    print(f"{tag:35s} {timing[tag]:7.1f}s", flush=True)


# 1) Validation croisée stratifiée sur le train (seed 0) — probas hors-pli
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for name in SUPERVISED:
    def oof(name=name):
        o = np.zeros(len(ybin), dtype=np.float32)
        for a, b in skf.split(Xtr, ymc):            # stratifié sur les 5 classes
            m = supervised(name, 0).fit(Xtr[a], ybin[a])
            o[b] = m.predict_proba(Xtr[b])[:, 1]
        return o
    run(f"oof_bin_{name}", oof)

# 2) Binaire et multi-classe, 5 graines
for s in SEEDS:
    for name in SUPERVISED:
        if name in ("LogReg",) and s > 0:          # déterministe : une seule graine suffit
            continue
        run(f"bin_{name}_s{s}", lambda: supervised(name, s).fit(Xtr, ybin).predict_proba(Xte)[:, 1])
        run(f"mc_{name}_s{s}", lambda: supervised(name, s).fit(Xtr, ymc).predict_proba(Xte))
        if name != "MLP":                          # MLPClassifier n'accepte pas class_weight
            run(f"mc_{name}_bal_s{s}",
                lambda: supervised(name, s, balanced=True).fit(Xtr, ymc).predict_proba(Xte))

# 3) Détection d'anomalies : apprise sur le trafic NORMAL seul (80 %),
#    les 20 % restants servent à fixer le seuil (budget de fausses alarmes)
Xn = Xtr[ybin == 0]
for s in SEEDS:
    Xn_fit, Xn_val = train_test_split(Xn, test_size=0.2, random_state=s)
    for det_name, Det in [("AE", AutoEncoderDetector), ("IForest", IForestDetector)]:
        if (P / f"ad_{det_name}_s{s}_test.npy").exists():
            continue
        t = time.time()
        d = Det(s).fit(Xn_fit)
        np.save(P / f"ad_{det_name}_s{s}_test.npy", d.score(Xte))
        np.save(P / f"ad_{det_name}_s{s}_valnorm.npy", d.score(Xn_val))
        np.save(P / f"ad_{det_name}_s{s}_train.npy", d.score(Xtr))
        timing[f"ad_{det_name}_s{s}"] = round(time.time() - t, 1)
        print(f"ad_{det_name}_s{s:<27} {timing[f'ad_{det_name}_s{s}']:7.1f}s", flush=True)

tp = ROOT / "results" / "timing.json"
old = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
old.update(timing)
tp.write_text(json.dumps(old, indent=1), encoding="utf-8")
print("TERMINÉ", flush=True)
