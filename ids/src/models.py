"""models.py — Modèles supervisés et détecteurs d'anomalies (semi-supervisés)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier

# Hyperparamètres fixés a priori (valeurs usuelles, documentées) puis vérifiés
# sur une validation interne au TRAIN — jamais sur le test.
def supervised(name: str, seed: int, balanced: bool = False):
    cw = "balanced" if balanced else None
    if name == "LogReg":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight=cw)
    if name == "DecisionTree":
        return DecisionTreeClassifier(min_samples_leaf=2, class_weight=cw, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=1, n_jobs=2,
                                      class_weight=cw, random_state=seed)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight=cw, random_state=seed, n_jobs=2, verbose=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
                             learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                             n_iter_no_change=6, random_state=seed)
    raise ValueError(name)


SUPERVISED = ["LogReg", "DecisionTree", "RandomForest", "LightGBM", "MLP"]


class AutoEncoderDetector:
    """Auto-encodeur entraîné sur le trafic NORMAL seul ; score = erreur de reconstruction."""
    def __init__(self, seed: int):
        self.m = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu",
                              batch_size=512, learning_rate_init=1e-3, max_iter=80,
                              early_stopping=True, n_iter_no_change=6, random_state=seed)

    def fit(self, X):
        self.m.fit(X, X)
        return self

    def score(self, X):
        return np.mean((self.m.predict(X) - X) ** 2, axis=1)


class IForestDetector:
    def __init__(self, seed: int):
        self.m = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=2)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)  # plus grand = plus anormal
