"""06_host_ablation_per_type.py — Effet, type par type, du retrait des variables host-based (LightGBM, graine 0)."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252)
from nslkdd import load, FEATURES, FEATURE_GROUPS, CATEGORICAL
from models import supervised

tr, te = load("train"), load("test")


def run(keep):
    num = [f for f in keep if f not in CATEGORICAL]
    ct = ColumnTransformer([("n", StandardScaler(), num),
                            ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    A, B = ct.fit_transform(tr[keep]), ct.transform(te[keep])
    return supervised("LightGBM", 0).fit(A, tr.is_attack).predict_proba(B)[:, 1] >= .5


host = FEATURE_GROUPS["host-based traffic (100 conn.)"]
d = te.assign(full=run(FEATURES), nohost=run([f for f in FEATURES if f not in host]))
g = (d[d.is_attack == 1].groupby("attack").agg(n=("full", "size"), avec=("full", "mean"), sans=("nohost", "mean")))
g["delta"] = g.sans - g.avec
g["inedit"] = ~g.index.isin(set(tr.attack))
out = Path(__file__).resolve().parents[1] / "results" / "host_ablation_per_type.csv"
g.sort_values("delta").to_csv(out)
print(g[g.n >= 50].sort_values("delta").round(2).to_string())
