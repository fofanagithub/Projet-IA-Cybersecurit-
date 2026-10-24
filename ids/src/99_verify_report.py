"""99_verify_report.py — Recalcule chaque chiffre cité dans le rapport à partir des sorties
et signale tout écart supérieur à l'arrondi affiché (ou à la tolérance numérique, cf. TOL_NUM)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
R = Path(__file__).resolve().parents[1] / "results"
eda = json.loads((R / "eda.json").read_text(encoding="utf-8"))
an = json.loads((R / "analysis.json").read_text(encoding="utf-8"))
ex = json.loads((R / "explain_ablation.json").read_text(encoding="utf-8"))
lao = json.loads((R / "lao_summary.json").read_text(encoding="utf-8"))
tb = pd.read_csv(R / "table_binary.csv")
tm = pd.read_csv(R / "table_multiclass.csv")
F1 = lambda m, s: 100 * tb[(tb["Modèle"] == m) & (tb["Évaluation"] == s)]["_F1"].iloc[0]
hy = an["anomaly_hybrid"]; op = an["operating_point_transfer"]; br = an["base_rate"]
cal = an["calibration"]; rsn = an["recall_seen_novel"]
CV = "CV 5 plis (train)"
M = ["LogReg", "DecisionTree", "RandomForest", "LightGBM", "MLP"]
# LogReg (lbfgs) et MLP (Adam) dépendent de l'algèbre linéaire sous-jacente (BLAS, nombre de
# threads, OS) : d'une machine à l'autre, leurs résultats varient de quelques dixièmes de point,
# alors que les modèles à arbres sont reproductibles au bit près. Les chiffres qui en dépendent
# (min/max sur les 5 modèles, MLP) reçoivent donc une tolérance plus large que l'arrondi affiché.
TOL_NUM = .5      # points de %
# Spearman sur 5 modèles : ρ ne prend que des multiples de 0,1 ; une seule inversion de rang
# entre deux modèles quasi ex aequo le décale de 0,1 — ces corrélations sont indicatives.
TOL_RANK = .1

claims = [  # (description, valeur recalculée, valeur citée, tolérance)
    ("part d'attaques inédites au test (%)", 100 * eda["novel_share_of_test_attacks"], 29.2, .05),
    ("AUC validation adversariale", eda["adversarial_validation_auc_all"], 0.910, .0005),
    ("AUC adversariale normal seul", eda["adversarial_validation_auc_normal_only"], 0.801, .0005),
    ("F1 LightGBM KDDTest+", F1("LightGBM", "KDDTest+"), 76.66, .005),
    ("F1 arbre KDDTest+", F1("DecisionTree", "KDDTest+"), 79.92, .005),
    ("écart CV-test min (pts)", min(F1(m, CV) - F1(m, "KDDTest+") for m in M), 19.9, TOL_NUM),
    ("écart CV-test max (pts)", max(F1(m, CV) - F1(m, "KDDTest+") for m in M), 25.3, TOL_NUM),
    ("écart CV-test21 min", min(F1(m, CV) - F1(m, "KDDTest-21") for m in M), 27.4, TOL_NUM),
    ("écart CV-test21 max", max(F1(m, CV) - F1(m, "KDDTest-21") for m in M), 36.4, TOL_NUM),
    ("Spearman CV vs test", an["spearman_cv_vs_test_rank"], 0.20, TOL_RANK),
    ("rappel inédits min (%)", 100 * min(rsn[m]["novel"][0] for m in M), 25.5, TOL_NUM),
    ("rappel inédits max (%)", 100 * max(rsn[m]["novel"][0] for m in M), 52.8, TOL_NUM),
    ("rappel vus min (%)", 100 * min(rsn[m]["seen"][0] for m in M), 75.5, TOL_NUM),
    ("rappel vus max (%)", 100 * max(rsn[m]["seen"][0] for m in M), 80.1, TOL_NUM),
    ("ECE LightGBM CV", cal["LightGBM"]["ECE_cv"], 0.0004, .00005),
    ("ECE LightGBM test", cal["LightGBM"]["ECE_test"], 0.21, .005),
    ("hybride LGBM+IF DR inédits (%)", 100 * hy["LightGBM+IForest"]["DR_novel"], 59.1, .05),
    ("hybride LGBM+IF FPR (%)", 100 * hy["LightGBM+IForest"]["FPR_test"], 3.07, .005),
    ("hybride LGBM+IF F1 (%)", 100 * hy["LightGBM+IForest"]["F1"], 82.7, .05),
    ("IForest FPR test (%)", 100 * hy["IForest"]["FPR_test"], 1.78, .005),
    ("AE DR à 1 % FPR (%)", 100 * hy["AE"]["DR"], 4.3, .05),
    ("LGBM seuil 0,1 % → FPR test (%)", 100 * op["LightGBM@0.001"]["FPR_test"], 3.01, .005),
    ("LGBM seuil 1 % → FPR test (%)", 100 * op["LightGBM@0.01"]["FPR_test"], 6.15, .005),
    ("précision à 1 % de prévalence (%)", 100 * br["LightGBM, seuil FPR 0,1 % (train)"]["precision@0.01"], 18.9, .05),
    ("précision à 10 % (%)", 100 * br["LightGBM, seuil FPR 0,1 % (train)"]["precision@0.1"], 72.0, .05),
    ("précision à 0,1 % (%)", 100 * br["LightGBM, seuil FPR 0,1 % (train)"]["precision@0.001"], 2.26, .005),
    ("SHAP part contenu (%)", 100 * ex["shap_by_group_share"]["content (contenu)"], 3.07, .005),
    ("ablation host-based DR inédits (%)", 100 * ex["ablation_lightgbm"]["host-based traffic (100 conn.)"]["DR_novel"], 56.1, .05),
    ("ablation host-based F1 (%)", 100 * ex["ablation_lightgbm"]["host-based traffic (100 conn.)"]["F1"], 81.0, .05),
    ("LAO Spearman vs test", lao["spearman_lao_vs_test_novel"], -0.10, TOL_RANK),
    ("F1 macro MLP", tm[(tm["Modèle"] == "MLP") & (tm["Évaluation"] == "KDDTest+")]["Macro-F1"].iloc[0], 0.609, TOL_NUM / 100),
]
bad = 0
for d, v, c, tol in claims:
    ok = abs(v - c) <= tol + 1e-9      # marge flottante (ρ = 0,1 calculé ≠ 0,1 exact)
    bad += not ok
    print(f"{'OK ' if ok else 'ÉCART'}  {d:40s} recalculé={v:.4f}  cité={c}  (tol. {tol:g})")
print(f"\n{len(claims) - bad}/{len(claims)} chiffres vérifiés")
