"""
nslkdd.py — Chargement, taxonomie des attaques et prétraitement NSL-KDD.

Principes de conception (anti-fuite de données) :
  * Toutes les statistiques de prétraitement (catégories one-hot, moyennes/écarts-types,
    sélection éventuelle) sont apprises UNIQUEMENT sur l'ensemble d'entraînement,
    via un sklearn.Pipeline, donc réappliquées à l'identique sur le test.
  * Les catégories inconnues au test (ex. services absents du train) sont gérées
    par handle_unknown="ignore" — aucune information du test ne remonte vers le train.
  * La colonne "difficulty" (niveau de difficulté NSL-KDD) n'est JAMAIS utilisée
    comme variable explicative : c'est une méta-donnée dérivée des prédictions
    de 21 classifieurs, l'utiliser serait une fuite de l'étiquette.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["attack", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# Variables à très forte asymétrie (volumes, compteurs) : transformation log1p
HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "hot", "num_failed_logins",
                "num_compromised", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "wrong_fragment", "urgent", "count", "srv_count",
                "dst_host_count", "dst_host_srv_count"]

# Groupes de variables (Tavallaee et al., 2009 ; Lee & Stolfo, 2000)
FEATURE_GROUPS = {
    "basic (intrinsèques)": FEATURES[:9],
    "content (contenu)": FEATURES[9:22],
    "time-based traffic (2 s)": FEATURES[22:31],
    "host-based traffic (100 conn.)": FEATURES[31:41],
}

# Taxonomie DARPA 1998/1999. Les attaques marquées (*) n'existent QUE dans le test.
ATTACK_CATEGORY = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load(split: str) -> pd.DataFrame:
    """split ∈ {'train', 'test', 'test21', 'train20'}"""
    fname = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt",
             "test21": "KDDTest-21.txt", "train20": "KDDTrain+_20Percent.txt"}[split]
    df = pd.read_csv(DATA_DIR / fname, header=None, names=COLUMNS)
    unknown = set(df["attack"]) - set(ATTACK_CATEGORY)
    if unknown:
        raise ValueError(f"Attaques non cartographiées : {unknown}")
    df["category"] = df["attack"].map(ATTACK_CATEGORY)
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def novel_mask(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Vrai pour les connexions du test dont le TYPE d'attaque n'apparaît pas au train."""
    seen = set(train["attack"])
    return (~test["attack"].isin(seen)) & (test["is_attack"] == 1)


def make_preprocessor(log_transform: bool = True, scale: bool = True) -> ColumnTransformer:
    num_steps = []
    if log_transform:
        heavy_idx = [NUMERIC.index(c) for c in HEAVY_TAILED]

        def _log(X):
            X = np.asarray(X, dtype=float).copy()
            X[:, heavy_idx] = np.log1p(np.clip(X[:, heavy_idx], 0, None))
            return X
        num_steps.append(("log1p", FunctionTransformer(_log, feature_names_out="one-to-one")))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"
    return ColumnTransformer(
        [("num", num_pipe, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        verbose_feature_names_out=False,
    )
