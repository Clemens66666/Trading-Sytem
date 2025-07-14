#!/usr/bin/env python3
# longtrend_ensemble_trainer_v2.py
# ------------------------------------------------------------------
"""
End-to-End-Trainer  •  Triple-Barrier-Labels  •  Anti-Leak-Filter
Bagging-&-Stacking-Ensemble aus
    LightGBM / XGBoost / CatBoost / ExtraTrees / 1D-CNN / LSTM / Transformer
Optuna-Tuning (SQLite-DB) • speichert finalen Ensemble-Pickle
"""
import warnings, json, joblib, optuna, gc
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

# ── classic ML ────────────────────────────────────────────────
import lightgbm as lgb
import xgboost  as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss



# ── deep-learning (TF/Keras) ──────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
tf.get_logger().setLevel("ERROR")               # mute TF verbosity
warnings.filterwarnings("ignore", category=FutureWarning)

# ── eigene Feature-Tools --------------------------------------
from features_utils import (
    load_ticks,
    make_hourly_bars,
    triple_barrier_label,
    leak_filter
)

# ═════════════ Konstanten ═════════════════════════════════════
RAW_FILE         = Path("E:/rawtickdata3.txt")
RAW_FILE_HOLDOUT = Path("E:/rawtickdatatestdata.txt")
MODEL_OUT        = Path("longtrend_ensemble_dlneu.pkl")
OPTUNA_DB        = "sqlite:///optuna_longtrend_ensemble_tb_dl.db"
CUTOFF_DATE      = pd.Timestamp("2025-01-01")
SEED             = 42
N_CLASSES        = 2              # feste Klassen-Anzahl (-1, 0, +1)

# ═════════════ DL-Hilfen ══════════════════════════════════════
def _compile(model: models.Model, lr: float) -> models.Model:
    model.compile(
        optimizer = optimizers.Adam(lr),
        loss      = losses.SparseCategoricalCrossentropy()
    )
    return model

def build_cnn(input_dim: int, hp: dict):
    return _compile(
        models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(hp["cnn_filters"], hp["cnn_k"], padding="same", activation="relu"),
            layers.MaxPool1D(2),
            layers.Conv1D(hp["cnn_filters"] * 2, hp["cnn_k"], padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(N_CLASSES, activation="softmax")
        ]),
        hp["dl_lr"]
    )

def build_lstm(input_dim: int, hp: dict):
    return _compile(
        models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.LSTM(hp["lstm_units"]),
            layers.Dense(32, activation="relu"),
            layers.Dense(N_CLASSES, activation="softmax")
        ]),
        hp["dl_lr"]
    )

def build_transformer(input_dim: int, hp: dict):
    inp = layers.Input(shape=(input_dim, 1))
    x   = layers.Conv1D(hp["tf_d_model"], 1, activation="linear")(inp)

    for _ in range(hp["tf_layers"]):
        # Self-Attention-Block
        attn = layers.MultiHeadAttention(num_heads=hp["tf_heads"],
                                         key_dim = hp["tf_d_model"])(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(hp["tf_dff"], activation="relu")(x)
        ff = layers.Dense(hp["tf_d_model"], activation="linear")(ff)
        x  = layers.Add()([x, ff])
        x  = layers.LayerNormalization()(x)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return _compile(models.Model(inp, out), hp["dl_lr"])

# ─────────── train_dl-Funktion für alle DL-Varianten ─────────────────
def train_dl(build_fn, X_tr, y_tr, X_va, y_va, hp):
    """
    Wrapper fürs Training der CNN / LSTM / Transformer. 
    build_fn(hp) liefert ein kompiliertes Keras-Model, der Rest kümmert sich 
    um DataFrame→NumPy, Reshapen und EarlyStopping.
    """
    # 1) DataFrame → NumPy
    X_tr_arr = X_tr.values if hasattr(X_tr, "values") else X_tr
    X_va_arr = X_va.values if hasattr(X_va, "values") else X_va

    # 2) Kanal-Dimension hinzufügen, falls nötig
    if X_tr_arr.ndim == 2:
        X_tr_arr = X_tr_arr[..., None]
        X_va_arr = X_va_arr[..., None]

    # 3) Model bauen (und kompilieren)
    model = build_fn(hp)

    # 4) EarlyStopping-Callback
    es = callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    # 5) Fit
    model.fit(
        X_tr_arr, y_tr,
        validation_data=(X_va_arr, y_va),
        epochs=100,
        batch_size=hp["dl_bs"],
        callbacks=[es],
        verbose=0
    )
    return model


# ════════════════════════════════════════════════════════════
#  DL-Training   –  Labels von {-1,0,+1}  →  {0,1,2} shiften
# ════════════════════════════════════════════════════════════

def fit_model(kind, X_tr, y_tr, X_va, y_va, p, seed):
    """
    Trainiert ein Basismodell vom Typ `kind` auf den Trainingsdaten.
    Für CNN/LSTM/Transformer verwenden wir `train_dl()`, dem wir
    einen passenden Builder übergeben.
    """

    # ─── LightGBM ────────────────────────────────────────────────────
    if kind == "lgb":
        cfg = dict(
            objective        = "multiclass",
            num_class        = N_CLASSES,
            metric           = "multi_logloss",
            learning_rate    = p["lr"],
            num_leaves       = p["leaves"],
            feature_fraction = p["feat_frac"],
            bagging_fraction = p["bag_frac"],
            bagging_freq     = p["bag_freq"],
            seed             = seed,
            verbose          = -1
        )
        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)
        return lgb.train(cfg, d_tr, 3000,
                         valid_sets=[d_va],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    # ─── XGBoost ────────────────────────────────────────────────────
    if kind == "xgb":
        cfg = {
            "objective":       "multi:softprob",
            "num_class":       N_CLASSES,
            "eta":              p["lr"],
            "max_depth":        p["xgb_depth"],
            "subsample":        p["bag_frac"],
            "colsample_bytree": p["feat_frac"],
            "seed":            seed,
            "verbosity":        0
        }
        d_tr = xgb.DMatrix(X_tr, label=y_tr)
        d_va = xgb.DMatrix(X_va, label=y_va)
        return xgb.train(cfg, d_tr, num_boost_round=3000,
                         evals=[(d_va, "va")],
                         early_stopping_rounds=50,
                         verbose_eval=False)

    # ─── CatBoost ───────────────────────────────────────────────────
    if kind == "cat":
        model = CatBoostClassifier(
            iterations   = 1000,
            learning_rate= p["lr"],
            depth        = p["cat_depth"],
            l2_leaf_reg  = p["cat_l2"],
            random_seed  = seed,
            verbose      = False
        )
        model.fit(X_tr, y_tr,
                  eval_set=(X_va, y_va),
                  early_stopping_rounds=50,
                  use_best_model=True)
        return model

    # ─── ExtraTrees ─────────────────────────────────────────────────
    if kind == "et":
        model = ExtraTreesClassifier(
            n_estimators = p["et_trees"],
            max_depth    = p["et_depth"],
            max_features = p["feat_frac"],
            random_state = seed,
            n_jobs       = -1
        )
        model.fit(X_tr, y_tr)
        return model

    # ─── Deep Learning ───────────────────────────────────────────────
    if kind in ("cnn", "lstm", "trans"):
        # bestimme Sequenzlänge
        seq_len = X_tr.shape[1]

        # baue für jede Variante einen 1-Arg-Builder, den train_dl() erwartet
        if kind == "cnn":
            builder_fn = lambda hp: build_cnn(seq_len, hp)
        elif kind == "lstm":
            builder_fn = lambda hp: build_lstm(seq_len, hp)
        else:  # "trans"
            builder_fn = lambda hp: build_transformer(seq_len, hp)

        # train_dl übernimmt DataFrame→NumPy, Reshape und ruft dann model = builder_fn(hp) auf
        return train_dl(builder_fn, X_tr, y_tr, X_va, y_va, p)

    # ─── alles andere ist ein Fehler ──────────────────────────────────
    raise ValueError(f"Unknown model kind: {kind}")


# ════════════════════════════════════════════════════════════
#  Probability-Wrapper  –  gibt immer 3-Spalten-Matrix zurück
# ════════════════════════════════════════════════════════════
def proba_dl(model, X):
    # 1) DataFrame -> numpy
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    # 2) Kanal-Dimension hinzufügen
    if X_arr.ndim == 2:
        X_arr = X_arr[..., None]
    return model.predict(X_arr, verbose=0)


# ════════════════════════════════════════════════════════════
#  predict_proba für alle Modell-Typen
# ════════════════════════════════════════════════════════════
def predict_proba(mdl, kind, X):
    if kind in ["lgb", "xgb", "cat", "et"]:
        # klassische Tree-Modelle liefern direkt proba
        return mdl.predict_proba(X)
    else:
        # alle Deep-Learning-Modelle gehen über proba_dl
        return proba_dl(mdl, X)


# ════════════════════════════════════════════════════════════
#  sicherstellen, dass es genau N_CLASSES Spalten sind
# ════════════════════════════════════════════════════════════
def ensure_prob_cols(p: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    if p.shape[1] == n_classes:
        return p
    out = np.zeros((p.shape[0], n_classes), dtype=p.dtype)
    out[:, : p.shape[1]] = p
    return out


# ════════════  ML-Model-Factory ─ Tree-Modelle ══════════════════
def _fit_tree(kind, X_tr, y_tr, X_va, y_va, p, seed):
    # ── Label-Shift:  (-1, 0, +1)  →  (0, 1, 2) ─────────────────
    y_tr = y_tr + 1
    y_va = y_va + 1

    # ─────────────────── LightGBM ───────────────────────────────
    if kind == "lgb":
        cfg = dict(objective        = "multiclass",
                   num_class        = N_CLASSES,
                   metric           = "multi_logloss",
                   learning_rate    = p["lr"],
                   num_leaves       = p["leaves"],
                   feature_fraction = p["feat_frac"],
                   bagging_fraction = p["bag_frac"],
                   bagging_freq     = p["bag_freq"],
                   seed             = seed,
                   verbose          = -1)
        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)
        return lgb.train(cfg, d_tr, 3000, valid_sets=[d_va],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    # ─────────────────── XGBoost ────────────────────────────────
    if kind == "xgb":
        cfg = dict(objective         = "multi:softprob",
                   num_class         = N_CLASSES,
                   eval_metric       = "mlogloss",
                   learning_rate     = p["lr"],
                   max_depth         = p["xgb_depth"],
                   subsample         = p["bag_frac"],
                   colsample_bytree  = p["feat_frac"],
                   seed              = seed)
        d_tr = xgb.DMatrix(X_tr, label=y_tr)
        d_va = xgb.DMatrix(X_va, label=y_va)
        return xgb.train(cfg, d_tr, 3000, evals=[(d_va, "va")],
                         early_stopping_rounds=80, verbose_eval=False)

    # ─────────────────── CatBoost ───────────────────────────────
    if kind == "cat":
        mdl = CatBoostClassifier(loss_function="MultiClass",
                                 learning_rate = p["lr"],
                                 depth         = p["cat_depth"],
                                 l2_leaf_reg   = p["cat_l2"],
                                 random_seed   = seed,
                                 verbose       = False)
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        return mdl

    # ─────────────────── ExtraTrees ─────────────────────────────
    if kind == "et":
        return ExtraTreesClassifier(n_estimators = p["et_trees"],
                                    max_depth    = p["et_depth"],
                                    max_features = p["feat_frac"],
                                    n_jobs       = -1,
                                    random_state = seed).fit(X_tr, y_tr)

    # ------------------------------------------------------------
    raise ValueError(f"Unknown tree model kind: {kind}")


def predict_proba(model, kind, X):
    if kind == "xgb":
        return model.predict(xgb.DMatrix(X))
    if kind == "lgb":
        return model.predict(X, num_iteration=model.best_iteration)
    if kind in ("cnn", "lstm", "trans"):
        return proba_dl(model, X)
    return model.predict_proba(X)

# ═════════════ Optuna-Objective ═══════════════════════════════

def objective(trial: optuna.trial.Trial) -> float:
    # ---- Label-Parameter ------------------------------------------------
    hor = trial.suggest_int("horizon_h", 24, 96, step=24)
    up  = trial.suggest_float("thr_up",    0.0001, 0.010)
    dn = trial.suggest_float("thr_dn",  0.0001, 0.010)

    # ---- Ensemble-Parameter ---------------------------------------------
    bag_n = trial.suggest_int("bag_n", 3, 6)
    kind  = trial.suggest_categorical("kind",
                ["lgb", "xgb", "cat", "et", "cnn", "lstm", "trans"])

    # ---- Modell-Hyper-Parameter -----------------------------------------
    hp = {
        # gemeinsame
        "lr"        : trial.suggest_float("lr", 1e-3, 0.3, log=True),
        "feat_frac" : trial.suggest_float("feat_frac", 0.5, 1.0),
        "bag_frac"  : trial.suggest_float("bag_frac", 0.5, 1.0),
        "bag_freq"  : trial.suggest_int  ("bag_freq", 1, 10),

        # LightGBM-/XGB-/Cat-/ET-spezifisch
        "leaves"    : trial.suggest_int  ("leaves", 20, 255),
        "xgb_depth" : trial.suggest_int  ("xgb_depth", 3, 9),
        "cat_depth" : trial.suggest_int  ("cat_depth", 3, 9),
        "cat_l2"    : trial.suggest_float("cat_l2", 1.0, 10.0),
        "et_trees"  : trial.suggest_int  ("et_trees", 100, 600, step=50),
        "et_depth"  : trial.suggest_int  ("et_depth", 3, 15),

        # Deep-Learning-Modelle
        "cnn_filters": trial.suggest_int ("cnn_filters", 8, 32),
        "cnn_k"     : trial.suggest_int  ("cnn_k", 2, 5),
        "lstm_units": trial.suggest_int  ("lstm_units", 16, 64),
        "tf_d_model": trial.suggest_int  ("tf_d_model", 16, 64),
        "tf_heads"  : trial.suggest_int  ("tf_heads", 2, 8),
        "tf_layers" : trial.suggest_int  ("tf_layers", 1, 3),
        "tf_dff"    : trial.suggest_int  ("tf_dff", 32, 128),
        "dl_bs"     : trial.suggest_categorical("dl_bs", [256, 512, 1024]),
        "dl_lr"     : trial.suggest_float("dl_lr", 1e-4, 1e-2, log=True),
    }  # <-- diese geschweifte Klammer schließt das Dict
    # ---- 1) Daten laden, Bars erzeugen ----------------------------------
    ticks = load_ticks(RAW_FILE)
    bars  = make_hourly_bars(ticks)

    # ---- 2) Labels berechnen --------------------------------------------
    bars["y"] = triple_barrier_label(bars, hor, up, dn)

    # ---- 3) Leak-Filter + NaNs entfernen, dann Neutral (1) rausschmeißen ---
    bars = leak_filter(bars, hor).dropna()
    bars = bars[bars["y"] != 1]      # alles außer Neutral (Label 1)

    # und direkt remappen auf 0/1 (statt -1/+1)
    bars = bars.copy()  # gegen SettingWithCopy-Warnung
    bars["y"] = bars["y"].map({0: 0, 2: 1})

    print(">>> Label-Verteilung nach Filter und Remap:")
    print(bars["y"].value_counts(), "\n")

    # ─── Globalen Label-Balance prüfen: mindestens 50 Samples pro Klasse ───
    lbl_counts = bars["y"].value_counts()
    if lbl_counts.get(0, 0) < 50 or lbl_counts.get(1, 0) < 50:
    # zu wenige Long- oder Short-Labels im gesamten Datensatz → skippen
     return -10.0

    # ---- 4) Features / Target splitten ----------------------------------
    y = bars["y"].astype(int)
    X = bars.drop(columns=["y", "TimeStamp"])

    # ---- 5) Zeit-Fenster und stratified Split ---------------------------
    cutoff_tr = CUTOFF_DATE - timedelta(days=90)
    train_mask = bars["TimeStamp"] < cutoff_tr

    X_tr_all, y_tr_all = X[train_mask], y[train_mask]
    X_hold,   y_hold   = X[~train_mask], y[~train_mask]

    # Stratified-Split im Trainingsbereich
    from sklearn.model_selection import train_test_split
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr_all, y_tr_all,
            test_size=0.25,
            random_state=SEED,
            stratify=y_tr_all
        )
    except ValueError:
        # Wenn stratify fehlschlägt (nur eine Klasse), dann Trial skippen
        return -10.0

    # ---- 6) Val-Klassen prüfen -------------------------------------------
    if len(np.unique(y_va)) < 2:
        return -10.0  # kein raise → Trial wird gespeichert, aber sehr schlecht bewertet

    # ---- NEU: Minimum an Negativ-Samples erzwingen --------------------
    n_neg_tr = int((y_tr == 0).sum())
    n_neg_va = int((y_va == 0).sum())
    if n_neg_tr < 50 or n_neg_va < 50:
        # zu wenig Negativ–Labels, Trial überspringen
        return -10.0

    # ---- 7) Bagging-Schleife (binary, zwingt 2 Klassen) ------------
    proba_va = None
    rng = np.random.default_rng(SEED)

    for b in range(bag_n):
    # Bootstrap-Sampling
        idx = rng.choice(len(X_tr), size=len(X_tr), replace=True)
        mdl = fit_model(kind,
                    X_tr.iloc[idx], y_tr.iloc[idx],
                    X_va,           y_va,
                    hp, SEED + b)

    # 1) rohe Wahrscheinlichkeiten holen
    p_b = predict_proba(mdl, kind, X_va)
    # 2) genau 2 Spalten erzwingen
    proba_b = ensure_prob_cols(p_b, n_classes=2)

    # initialisiere proba_va beim ersten Durchlauf
    if proba_va is None:
        proba_va = np.zeros_like(proba_b)

    proba_va += proba_b / bag_n

    del mdl, p_b, proba_b
    gc.collect()

    # ---- 8) Meta-Regressor -----------------------------------------------
    meta = LogisticRegression(
        C=trial.suggest_float("meta_C", 0.1, 10, log=True),
        penalty=trial.suggest_categorical("meta_pen", ["l2", "none"]),
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced"
    )
    meta.fit(proba_va, y_va)
    final = meta.predict_proba(proba_va)  # shape (n_samples, 2)

    # ---- 9) Log-Loss Score berechnen -------------------------------------
    score = -log_loss(y_va, final)


    # ---- 10) Trade-Anzahl-Penalty ----------------------------------------
    n_tr = int((y_tr_all == 1).sum() + (y_tr_all == -1).sum())
    if n_tr < 10:
        score -= 0.05
    trial.set_user_attr("n_trades", n_tr)

    return score

# ═════════════ Retrain mit besten Param ═══════════════════════
def retrain(best):
    lp    = best.params
    ticks = load_ticks(RAW_FILE)
    bars  = make_hourly_bars(ticks)
    bars["y"] = triple_barrier_label(
                    bars, lp["horizon_h"], lp["thr_up"], lp["thr_dn"])
    bars  = leak_filter(bars, lp["horizon_h"]).dropna()

    # 1) neutrales Label (1) rauswerfen
    bars = bars[bars["y"] != 0].copy()

    # 2) auf binary remappen: 0→0 (short), 2→1 (long)
    bars["y"] = bars["y"].map({1: 1, 2: 0}).astype(int)

    # 3) jetzt X und y erzeugen
    y = bars.pop("y").to_numpy()
    X = bars.drop(columns=["TimeStamp"]).to_numpy()


    kind  = lp["kind"]
    hp    = lp.copy()
    bag_n = lp["bag_n"]

    bases = []
    for s in range(bag_n):
        mdl = fit_model(kind, X, y, X, y, hp, SEED + s)
        bases.append(mdl)

    all_p = np.zeros((len(X), N_CLASSES))
    for mdl in bases:
        all_p += ensure_prob_cols(predict_proba(mdl, kind, X)) / bag_n

    meta = LogisticRegression(
            C           = lp["meta_C"],
            penalty     = lp["meta_pen"],
            multi_class = "multinomial",
            solver      = "lbfgs",
            max_iter    = 1000
    ).fit(all_p, y)

    joblib.dump(dict(
        base_kind    = kind,
        base_models  = bases,
        meta         = meta,
        feat_cols    = list(bars.columns),
        label_params = dict(horizon_h = lp["horizon_h"],
                            thr_up     = lp["thr_up"],
                            thr_dn     = lp["thr_dn"])
    ), MODEL_OUT, compress=3)
    print("✅  Modell gespeichert:", MODEL_OUT)

# ═════════════ main ═══════════════════════════════════════════
def main():
    study = optuna.create_study(
                storage      = OPTUNA_DB,
                study_name   = "ensemble_dl11",
                direction    = "maximize",
                load_if_exists = True
    )
    study.optimize(objective, n_trials=1)

    print("\nBest Score:", study.best_value)
    print(json.dumps(study.best_params, indent=2))
    retrain(study.best_trial)

import joblib
from sklearn.metrics import log_loss, roc_auc_score, classification_report

def evaluate_holdout(model_path, raw_holdout_path):
    # 1) Modell und Meta-Regressor laden
    ens = joblib.load(model_path)
    bases      = ens["base_models"]
    meta       = ens["meta"]
    feat_cols  = ens["feat_cols"]
    lp         = ens["label_params"]     # {'horizon_h':…, 'thr_up':…, 'thr_dn':…}
    kind       = ens["base_kind"]
    bag_n      = len(bases)

    # 2) Hold-Out Daten laden & Bars bauen
    ticks_hold = load_ticks(Path(raw_holdout_path))
    bars_hold  = make_hourly_bars(ticks_hold)

    # 3) Labels mit gleichen Parametern erzeugen
    bars_hold["y"] = triple_barrier_label(
        bars_hold,
        lp["horizon_h"],
        lp["thr_up"],
        lp["thr_dn"]
    )

    # 4) Leak-Filter & NaNs
    bars_hold = leak_filter(bars_hold, lp["horizon_h"]).dropna()

    # … nach leak_filter & dropna(), vor bars_hold[bars_hold["y"] != 1] …
    raw_labels = bars_hold["y"].value_counts(dropna=False)
    print("Hold-Out raw label counts (−1/0/+1):", raw_labels.to_dict())


    # 5) Neutral raus, remappen auf 0/1
    bars_hold = bars_hold[bars_hold["y"] != 0].copy()
    bars_hold["y"] = bars_hold["y"].map({1: 1, 2: 0}).astype(int)

    # 6) Feature-Matrix und Target
    y_hold = bars_hold["y"].to_numpy()
    # sicherstellen, dass TimeStamp nicht dabei ist
    feat_cols_clean = [c for c in feat_cols if c in bars_hold.columns and c != "TimeStamp"]
    X_hold = bars_hold[feat_cols_clean].to_numpy()

    # 7) Basis-Prognosen mitteln
    proba_hold = np.zeros((len(y_hold), meta.coef_.shape[1]))
    for mdl in bases:
        p = predict_proba(mdl, kind, X_hold)
        proba_hold += ensure_prob_cols(p) / bag_n

    # 8) Meta-Regressor anwenden
    final_proba = meta.predict_proba(proba_hold)
    final_pred  = final_proba.argmax(axis=1)

    # 9) Kennzahlen berechnen
    ll  = log_loss(y_hold, final_proba)
    auc = roc_auc_score(y_hold, final_proba[:, 1])  # binäre AUC
    print("=== Hold-Out Evaluation ===")
    print(f"Log-Loss: {ll:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\nClassification Report:\n",
          classification_report(y_hold, final_pred, digits=4))

# Am Ende in main():
if __name__ == "__main__":
    main()
    evaluate_holdout(MODEL_OUT, RAW_FILE_HOLDOUT)
