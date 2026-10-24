"""03_analysis.py — Métriques, intervalles de confiance, tests statistiques, figures.

Toutes les décisions (seuils, choix de modèle) sont prises sur des données
d'entraînement (probas hors-pli ou validation interne) puis TRANSFÉRÉES au test.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, novel_mask, CLASSES
from models import SUPERVISED

ROOT = Path(__file__).resolve().parents[1]
P, FIG = ROOT / "results" / "preds", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
RNG = np.random.default_rng(2026)

# ---------- style des figures (palette validée, catégorielle en ordre fixe)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {m: C[i] for i, m in enumerate(SUPERVISED)}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
})

tr, te = load("train"), load("test")
y = te["is_attack"].values
ymc = te["category"].map({c: i for i, c in enumerate(CLASSES)}).values
ytr = tr["is_attack"].values
M_ALL = np.ones(len(te), bool)
M_21 = (te["difficulty"] < 21).values
NOVEL = novel_mask(tr, te).values
SEEN = (y == 1) & ~NOVEL
NORMAL = y == 0
R = {}


def seeds_of(m):
    return [0] if m == "LogReg" else SEEDS


def binm(yt, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(yt, yp), "Precision": precision_score(yt, yp, zero_division=0),
            "Recall (DR)": recall_score(yt, yp), "F1": f1_score(yt, yp),
            "FPR": fp / (fp + tn), "MCC": matthews_corrcoef(yt, yp),
            "ROC-AUC": roc_auc_score(yt, p), "PR-AUC": average_precision_score(yt, p)}


def agg(rows):
    df = pd.DataFrame(rows)
    return df.mean(), df.std(ddof=1) if len(df) > 1 else df.mean() * 0


# =====================================================================
# A) Binaire : CV (train) vs KDDTest+ vs KDDTest-21 ; vu vs inédit
# =====================================================================
tabA = []
for m in SUPERVISED:
    oof = np.load(P / f"oof_bin_{m}.npy")
    cv = binm(ytr, oof)
    for split, mask in [("CV 5 plis (train)", None), ("KDDTest+", M_ALL), ("KDDTest-21", M_21)]:
        if mask is None:
            mu, sd = pd.Series(cv), pd.Series(cv) * 0
        else:
            mu, sd = agg([binm(y[mask], np.load(P / f"bin_{m}_s{s}.npy")[mask]) for s in seeds_of(m)])
        tabA.append({"Modèle": m, "Évaluation": split,
                     **{k: f"{100*mu[k]:.2f} ± {100*sd[k]:.2f}" if split != "CV 5 plis (train)" else f"{100*mu[k]:.2f}"
                        for k in mu.index},
                     **{f"_{k}": mu[k] for k in mu.index}})
    # rappel sur attaques vues / inédites
    rs = [(np.load(P / f"bin_{m}_s{s}.npy") >= .5) for s in seeds_of(m)]
    R.setdefault("recall_seen_novel", {})[m] = {
        "seen": [float(np.mean([r[SEEN].mean() for r in rs])), float(np.std([r[SEEN].mean() for r in rs]))],
        "novel": [float(np.mean([r[NOVEL].mean() for r in rs])), float(np.std([r[NOVEL].mean() for r in rs]))],
        "fpr_test": float(np.mean([r[NORMAL].mean() for r in rs]))}
dfA = pd.DataFrame(tabA)
dfA.to_csv(ROOT / "results" / "table_binary.csv", index=False)
print(dfA[[c for c in dfA.columns if not c.startswith("_")]].to_string(index=False))

# Choix du « meilleur » modèle : sur la CV du TRAIN (protocole honnête) ET, pour info, sur le test
cvF1 = dfA[dfA["Évaluation"].str.startswith("CV")].set_index("Modèle")["_F1"]
teF1 = dfA[dfA["Évaluation"] == "KDDTest+"].set_index("Modèle")["_F1"]
R["best_by_cv"] = cvF1.idxmax()
R["best_by_test"] = teF1.idxmax()
R["spearman_cv_vs_test_rank"] = float(pd.Series(cvF1).rank().corr(pd.Series(teF1).rank(), method="spearman"))

# =====================================================================
# B) IC bootstrap (graine 0) + McNemar exact avec correction de Holm
# =====================================================================
B = 1000
idx = RNG.integers(0, len(y), size=(B, len(y)))
ci = {}
for m in SUPERVISED:
    p = np.load(P / f"bin_{m}_s0.npy")
    yp = (p >= .5).astype(int)
    f1s = [f1_score(y[i], yp[i]) for i in idx[:300]]       # 300 suffisent pour F1 (coûteux)
    aucs = [roc_auc_score(y[i], p[i]) for i in idx[:300]]
    ci[m] = {"F1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
             "AUC": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
R["bootstrap_ci_seed0"] = ci

ref = R["best_by_cv"]           # référence choisie sur le train : stable, et sans regarder le test
pref = (np.load(P / f"bin_{ref}_s0.npy") >= .5) == y
mc = {}
for m in SUPERVISED:
    if m == ref:
        continue
    pm = (np.load(P / f"bin_{m}_s0.npy") >= .5) == y
    b, c = int((pref & ~pm).sum()), int((~pref & pm).sum())
    mc[m] = {"ref_right_other_wrong": b, "ref_wrong_other_right": c,
             "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}
# Holm
order = sorted(mc, key=lambda k: mc[k]["p"])
for i, k in enumerate(order):
    mc[k]["p_holm"] = min(1.0, max(mc[o]["p"] * (len(order) - j) for j, o in enumerate(order[:i + 1])))
R["mcnemar_vs_" + ref] = mc

# =====================================================================
# C) Multi-classe (5 classes) : pondération des classes ou non
# =====================================================================
tabC, per_class = [], {}
for m in SUPERVISED:
    for bal in ([False, True] if m != "MLP" else [False]):
        tag = f"mc_{m}{'_bal' if bal else ''}"
        rows = []
        for s in seeds_of(m):
            pr = np.load(P / f"{tag}_s{s}.npy").argmax(1)
            for name, mask in [("KDDTest+", M_ALL), ("KDDTest-21", M_21)]:
                rec = recall_score(ymc[mask], pr[mask], labels=range(5), average=None, zero_division=0)
                f1c = f1_score(ymc[mask], pr[mask], labels=range(5), average=None, zero_division=0)
                rows.append({"split": name, "seed": s, "Accuracy": accuracy_score(ymc[mask], pr[mask]),
                             "Macro-F1": f1_score(ymc[mask], pr[mask], average="macro"),
                             "Weighted-F1": f1_score(ymc[mask], pr[mask], average="weighted"),
                             **{f"Rappel {c}": rec[i] for i, c in enumerate(CLASSES)},
                             **{f"F1 {c}": f1c[i] for i, c in enumerate(CLASSES)}})
        d = pd.DataFrame(rows)
        for split, g in d.groupby("split"):
            mu, sd = g.drop(columns=["split", "seed"]).mean(), g.drop(columns=["split", "seed"]).std(ddof=1)
            tabC.append({"Modèle": m + (" (pondéré)" if bal else ""), "Évaluation": split,
                         **{k: mu[k] for k in mu.index}, **{f"sd_{k}": (sd[k] if len(g) > 1 else 0) for k in mu.index}})
dfC = pd.DataFrame(tabC)
dfC.to_csv(ROOT / "results" / "table_multiclass.csv", index=False)
print(dfC[dfC["Évaluation"] == "KDDTest+"][["Modèle", "Accuracy", "Macro-F1", "sd_Macro-F1"] +
      [f"Rappel {c}" for c in CLASSES]].round(4).to_string(index=False))

# =====================================================================
# D) Détection d'anomalies + détecteur hybride (budget 1 % FPR fixé sur validation normale)
# =====================================================================
hyb = {}
for det in ["AE", "IForest"]:
    rows = []
    for s in SEEDS:
        sc, vn = np.load(P / f"ad_{det}_s{s}_test.npy"), np.load(P / f"ad_{det}_s{s}_valnorm.npy")
        thr = np.quantile(vn, 0.99)
        a = sc >= thr
        rows.append({"AUC": roc_auc_score(y, sc), "DR": a[y == 1].mean(), "FPR_test": a[NORMAL].mean(),
                     "DR_seen": a[SEEN].mean(), "DR_novel": a[NOVEL].mean()})
    hyb[det] = pd.DataFrame(rows).mean().to_dict()
# hybride : supervisé (seuil 0,5) OU anomalie (seuil 1 % FPR sur validation)
for m in ["RandomForest", "LightGBM", "MLP"]:
    for det in ["AE", "IForest"]:
        rows = []
        for s in SEEDS:
            sup = np.load(P / f"bin_{m}_s{s}.npy") >= .5
            sc, vn = np.load(P / f"ad_{det}_s{s}_test.npy"), np.load(P / f"ad_{det}_s{s}_valnorm.npy")
            a = sup | (sc >= np.quantile(vn, 0.99))
            rows.append({"DR": a[y == 1].mean(), "FPR_test": a[NORMAL].mean(), "DR_seen": a[SEEN].mean(),
                         "DR_novel": a[NOVEL].mean(), "F1": f1_score(y, a),
                         "sup_DR_novel": sup[NOVEL].mean(), "sup_FPR": sup[NORMAL].mean(), "sup_F1": f1_score(y, sup)})
        hyb[f"{m}+{det}"] = pd.DataFrame(rows).mean().to_dict()
R["anomaly_hybrid"] = hyb

# =====================================================================
# E) Transfert du point de fonctionnement et taux de base (base-rate fallacy)
# =====================================================================
op = {}
for m in SUPERVISED:
    oof = np.load(P / f"oof_bin_{m}.npy")
    for target in [0.01, 0.001]:
        on = oof[ytr == 0]
        thr = np.quantile(on, 1 - target)                     # seuil choisi sur le TRAIN
        if (on >= thr).mean() > target:                       # ex-aequo (scores discrets, ex. arbre) :
            above = np.unique(on[on > thr])                   # on monte au score distinct suivant
            thr = above[0] if len(above) else np.nextafter(thr, 2)
        rows = []
        for s in seeds_of(m):
            p = np.load(P / f"bin_{m}_s{s}.npy")
            rows.append({"thr": thr, "FPR_train_oof": (oof[ytr == 0] >= thr).mean(),
                         "FPR_test": (p[NORMAL] >= thr).mean(), "DR_test": (p[y == 1] >= thr).mean()})
        op[f"{m}@{target}"] = pd.DataFrame(rows).mean().to_dict()
R["operating_point_transfer"] = op


def precision_at_prevalence(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


# =====================================================================
# F) Calibration (ECE, Brier) du binaire
# =====================================================================
def ece(yt, p, n=15):
    bins = np.linspace(0, 1, n + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if k.any():
            e += k.mean() * abs(yt[k].mean() - p[k].mean())
    return e


R["calibration"] = {m: {"ECE_test": float(np.mean([ece(y, np.load(P / f"bin_{m}_s{s}.npy")) for s in seeds_of(m)])),
                        "Brier_test": float(np.mean([np.mean((np.load(P / f"bin_{m}_s{s}.npy") - y) ** 2) for s in seeds_of(m)])),
                        "ECE_cv": float(ece(ytr, np.load(P / f"oof_bin_{m}.npy")))} for m in SUPERVISED}

# =====================================================================
# G) Rappel par type d'attaque (meilleur modèle, moyenne des graines)
# =====================================================================
best = R["best_by_cv"]          # modèle RETENU = choisi sur le train (jamais sur le test)
pb =np.mean([np.load(P / f"bin_{best}_s{s}.npy") >= .5 for s in seeds_of(best)], axis=0)
seen_types = set(tr["attack"])
pt = (te.assign(det=pb)[te.is_attack == 1].groupby("attack")
      .agg(n=("det", "size"), recall=("det", "mean"), category=("category", "first")).reset_index())
pt["novel"] = ~pt["attack"].isin(seen_types)
pt = pt.sort_values("recall")
pt.to_csv(ROOT / "results" / "per_attack_recall.csv", index=False)

(ROOT / "results" / "analysis.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# =====================================================================
# FIGURES
# =====================================================================
# Fig 1 — décalage de priors entre train et test
fig, ax = plt.subplots(figsize=(7, 3.4))
x = np.arange(len(CLASSES)); w = 0.38
ptr = tr["category"].value_counts(normalize=True).reindex(CLASSES) * 100
pte = te["category"].value_counts(normalize=True).reindex(CLASSES) * 100
b1 = ax.bar(x - w / 2 - .01, ptr, w, color=C[0], label="KDDTrain+")
b2 = ax.bar(x + w / 2 + .01, pte, w, color=C[1], label="KDDTest+")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8, color=INK2, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(x, CLASSES); ax.set_ylabel("% des connexions"); ax.grid(axis="x", visible=False)
ax.set_title("Répartition des classes : le test sur-représente fortement R2L (×16)")
ax.legend(loc="upper right")
fig.savefig(FIG / "fig1_class_shift.png"); plt.close(fig)

# Fig 2 — l'illusion de la validation croisée
fig, ax = plt.subplots(figsize=(7, 3.6))
splits = ["CV 5 plis (train)", "KDDTest+", "KDDTest-21"]
mk = {"CV 5 plis (train)": "o", "KDDTest+": "s", "KDDTest-21": "D"}
sc_ = {"CV 5 plis (train)": C[0], "KDDTest+": C[1], "KDDTest-21": C[2]}
for i, m in enumerate(SUPERVISED):
    vals = [dfA[(dfA["Modèle"] == m) & (dfA["Évaluation"] == s)]["_F1"].iloc[0] * 100 for s in splits]
    ax.plot([i, i], [min(vals), max(vals)], color=GRID, lw=3, zorder=1)
    for s, v in zip(splits, vals):
        ax.scatter(i, v, s=60, marker=mk[s], color=sc_[s], edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=s if i == 0 else None)
        ax.annotate(f"{v:.1f}", (i, v), xytext=(9, -3), textcoords="offset points", fontsize=8, color=INK2)
ax.set_xticks(range(len(SUPERVISED)), SUPERVISED); ax.set_ylabel("F1 (%) – classe attaque")
ax.set_title("Même modèle, trois évaluations : la CV aléatoire surestime massivement")
ax.legend(loc="lower left", ncol=3); ax.set_ylim(40, 102); ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig2_cv_vs_test.png"); plt.close(fig)

# Fig 3 — rappel attaques vues vs inédites
fig, ax = plt.subplots(figsize=(7, 3.4))
rsn = R["recall_seen_novel"]
for k, (key, col, lab) in enumerate([("seen", C[0], "Types vus à l'entraînement"),
                                     ("novel", C[1], "Types inédits (zero-day simulé)")]):
    vals = [rsn[m][key][0] * 100 for m in SUPERVISED]
    err = [rsn[m][key][1] * 100 for m in SUPERVISED]
    bars = ax.bar(np.arange(5) + (k - .5) * .4, vals, .38, yerr=err, color=col, label=lab,
                  error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=8, color=INK2, xytext=(0, 3), textcoords="offset points")
ax.set_xticks(range(5), SUPERVISED); ax.set_ylabel("Taux de détection (%)"); ax.set_ylim(0, 105)
ax.set_title("Détection des attaques connues vs inédites (KDDTest+, seuil 0,5)")
ax.legend(loc="upper right", ncol=1, fontsize=8); ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig3_seen_vs_novel.png"); plt.close(fig)

# Fig 4 — ROC sur KDDTest+ (graine 0), zoom faible FPR, échelle log
fig, ax = plt.subplots(figsize=(6, 4))
for m in SUPERVISED:
    p = np.load(P / f"bin_{m}_s0.npy")
    f, t, _ = roc_curve(y, p)
    ax.plot(f, t, color=MODEL_COLOR[m], label=f"{m} (AUC {roc_auc_score(y, p):.3f})")
for det, col in [("AE", C[5]), ("IForest", C[6])]:
    sc = np.load(P / f"ad_{det}_s0_test.npy")
    f, t, _ = roc_curve(y, sc)
    ax.plot(f, t, color=col, ls="--", label=f"{det} – normal seul (AUC {roc_auc_score(y, sc):.3f})")
ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("Taux de faux positifs (échelle log)"); ax.set_ylabel("Taux de détection")
ax.set_title("Courbes ROC sur KDDTest+ (FPR en échelle log)")
ax.axvline(0.01, color=INK2, lw=.8, ls=":"); ax.text(0.0105, 0.95, "FPR = 1 %", fontsize=8, color=INK2)
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.savefig(FIG / "fig4_roc.png"); plt.close(fig)

# Fig 5 — matrice de confusion multi-classe normalisée (meilleur modèle macro-F1)
mcF = dfC[dfC["Évaluation"] == "KDDTest+"].set_index("Modèle")["Macro-F1"]
best_mc = mcF.idxmax()
R["best_multiclass"] = best_mc
tag = "mc_" + best_mc.replace(" (pondéré)", "") + ("_bal" if "pondéré" in best_mc else "")
pr = np.load(P / f"{tag}_s0.npy").argmax(1)
cm = confusion_matrix(ymc, pr, labels=range(5), normalize="true") * 100
fig, ax = plt.subplots(figsize=(5, 4.2))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
ax.imshow(cm, cmap=cmap, vmin=0, vmax=100)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > 55 else INK)
ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES); ax.grid(False)
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
ax.set_title(f"Confusion (% par ligne) — {best_mc}")
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(FIG / "fig5_confusion.png"); plt.close(fig)

# Fig 6 — rappel par type d'attaque
fig, ax = plt.subplots(figsize=(7, 8))
cols = [C[1] if n else C[0] for n in pt["novel"]]
ax.barh(range(len(pt)), pt["recall"] * 100, color=cols, height=.7)
ax.set_yticks(range(len(pt)), [f"{a} ({c}, n={n})" for a, c, n in zip(pt.attack, pt.category, pt.n)], fontsize=7.5)
ax.set_xlabel("Taux de détection (%)"); ax.set_xlim(0, 105); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C[0], label="type vu à l'entraînement"),
                   Patch(color=C[1], label="type inédit")], loc="lower right")
ax.set_title(f"Détection par type d'attaque — {best} (moyenne 5 graines)")
fig.savefig(FIG / "fig6_per_attack.png"); plt.close(fig)

# Fig 7 — base-rate fallacy : ce que la CV promettait vs ce que le test révèle
o = R["operating_point_transfer"]
for m, lab in [(best, f"{best}, seuil 0,5"), (best + "@0.01", f"{best}, seuil FPR 1 % (train)"),
               (best + "@0.001", f"{best}, seuil FPR 0,1 % (train)")]:
    if "@" in m:
        tpr, fpr = o[m]["DR_test"], o[m]["FPR_test"]
    else:
        tpr = float(np.mean([(np.load(P / f"bin_{best}_s{s}.npy")[y == 1] >= .5).mean() for s in seeds_of(best)]))
        fpr = R["recall_seen_novel"][best]["fpr_test"]
    R.setdefault("base_rate", {})[lab] = {"DR": tpr, "FPR": fpr, **{
        f"precision@{pi}": float(precision_at_prevalence(tpr, fpr, pi)) for pi in [0.1, 0.01, 0.001]},
        "fausses_alertes_par_million_conn": fpr * 1e6}
k = best + "@0.001"
dr, fpr_cv, fpr_te = o[k]["DR_test"], o[k]["FPR_train_oof"], o[k]["FPR_test"]
fig, ax = plt.subplots(figsize=(7, 3.8))
pis = np.logspace(-3, 1.8, 300) / 100                  # 0,001 % → 63 %, au-delà de la prévalence du test
ax.plot(pis * 100, precision_at_prevalence(dr, fpr_cv, pis) * 100, color=GRID, lw=2.5,
        label=f"Promis par la CV : FPR {fpr_cv*100:.1f} %".replace(".", ","))
ax.plot(pis * 100, precision_at_prevalence(dr, fpr_te, pis) * 100, color=C[0],
        label=f"Mesuré sur KDDTest+ : FPR {fpr_te*100:.1f} %".replace(".", ","))
for pi in [0.01, 0.001]:
    v = precision_at_prevalence(dr, fpr_te, pi) * 100
    ax.scatter(pi * 100, v, s=36, color=C[0], zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.annotate(f"{v:.0f} % des alertes\nsont vraies à π = {pi*100:g} %".replace(".", ","), (pi * 100, v), xytext=(12, 4),
                textcoords="offset points", fontsize=8, color=INK)
prev = y.mean() * 100                                   # part des ATTAQUES dans KDDTest+
ax.axvline(prev, color=INK2, lw=.8, ls=":")
ax.text(prev * .93, 55, f"prévalence\nde KDDTest+\n({prev:.1f} %)".replace(".", ","), fontsize=8, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_xlabel("Prévalence π des attaques dans le trafic (%) — échelle log")
ax.set_ylabel("Précision des alertes (%)"); ax.set_ylim(0, 102)
ax.set_title(f"{best}, DR {dr*100:.0f} % : un FPR 30× plus élevé que prévu ruine la précision")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(FIG / "fig7_base_rate.png"); plt.close(fig)

(ROOT / "results" / "analysis.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in R.items() if k not in ("bootstrap_ci_seed0",)}, indent=1, default=float))
