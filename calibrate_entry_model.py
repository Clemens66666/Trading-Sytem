#!/usr/bin/env python3
# entry_refit_from_optuna.py
# Re-fit des Entry-Modells MIT den Optuna-Best-Params (kein Re-Tuning)
# ───────────────────────────────────────────────────────────────────
import joblib, logging, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.ensemble     import (RandomForestClassifier,
                                  HistGradientBoostingClassifier,
                                  StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration  import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics      import (accuracy_score, classification_report,
                                  confusion_matrix, f1_score, log_loss,
                                  balanced_accuracy_score)
import optuna

from xgboost   import XGBClassifier
from lightgbm  import LGBMClassifier
from catboost  import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve     

# ─── Helper aus deinem Haupt-Trainer ──────────────────────────────
from features_utils import make_hourly_bars
from entry_model_trainer import (compute_trend_probabilities,
                                 prepare_10min_bars,
                                 build_feature_df_chunks,
                                 label_high_low_df,
                                 resample_to_bars, ensure_timestamp,
                                 standardize_tick_last)

# ─── Konstanten / Dateien ─────────────────────────────────────────
DATA_RAW     = "E:/RawTickData3.txt"
LONG_MODEL   = Path("E:/longtrend_ensemble_dl.pkl")
OPTUNA_DB    = "sqlite:///optuna.db"
OUT_PATH     = "E:/entry_stack_fullopt1min.pkl"

BAR_RULE_MIN = "1T"
BAR_RULE_HR  = "1H"
KFOLD        = 6
EPS          = 1e-6
N_CLASSES    = 2     # 0 = LOW/Long, 1 = HIGH/Short

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
warnings.filterwarnings("ignore")

# ═══════════ 1) Optuna-Best-Parameter laden ═══════════════════════
def best_params(study_name: str):
    st = optuna.load_study(study_name=study_name, storage=OPTUNA_DB)
    return st.best_trial.params

par_sent   = best_params("sent_chunk10")        # {'long_window': 442, ...}
par_feat   = best_params("feat_chunk10")        # vol/rsi/ofi usw.
win_label  = best_params("label_zone8")["window_minutes"]   # int
meta_C     = best_params("meta_logreg12")["C"]

# Basis-Model-Params
bm_xgb = best_params("base13_xgb")
bm_lgb = best_params("base13_lgb")
bm_cat = best_params("base13_cat")
bm_rf  = best_params("base13_rf")
bm_ebm = best_params("base13_ebm")
bm_tab = best_params("base13_tabnet")

# ═══════════ 2) Daten-Vorverarbeitung ═════════════════════════════
logging.info("⇢ Starte Re-Fit mit gespeicherten Parametern")

# Trend-Ensemble → Stunden-Probas
ens   = joblib.load(LONG_MODEL)
hours, trend_ser = compute_trend_probabilities(
        ens["base_models"], ens["meta"], ens["feat_cols"])

# 10-Min-Bars / Features
bars_10m   = prepare_10min_bars(hours, trend_ser)
feat_raw   = build_feature_df_chunks(par_sent, par_feat)
feat       = feat_raw.merge(
                 bars_10m[["TimeStamp","trend_prob_long"]],
                 on="TimeStamp", how="left")

# Labels (High/Low)
 # --- Labels zu Integer umwandeln & Spalten harmonisieren ----------
labels = label_high_low_df(
            resample_to_bars(bars_10m, BAR_RULE_MIN),
            window_minutes=win_label)

labels = (labels
          .assign(entry_label = labels["zone_type"].map(
                    {"LOW_ZONE":0, "HIGH_ZONE":1}))
          .rename(columns={"timestamp":"TimeStamp"})
          [["TimeStamp","entry_label"]])          # nur 2 Spalten!

feat = feat.merge(labels, on="TimeStamp", how="inner")

y      = feat["entry_label"].astype(int)
X      = (feat
           .drop(columns=["entry_label","Tick_Last","Tick_Volume"],
                 errors="ignore")
           .fillna(0.0))

# ═══════════ 3) Basis-Modelle exakt mit Best-Params ═══════════════
models = {
    "xgb": XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         learning_rate=bm_xgb["xgb_lr"],
                         max_depth=bm_xgb["xgb_md"],
                         n_estimators=bm_xgb["xgb_ne"],
                         subsample=bm_xgb["xgb_ss"], n_jobs=2),
    "lgb": LGBMClassifier(objective="binary",
                         learning_rate=bm_lgb["lgb_lr"],
                         max_depth=bm_lgb["lgb_md"],
                         n_estimators=bm_lgb["lgb_ne"],
                         subsample=bm_lgb["lgb_ss"], n_jobs=2),
    "cat": CatBoostClassifier(loss_function="Logloss", verbose=False,
                         learning_rate=bm_cat["cat_lr"],
                         depth=bm_cat["cat_md"],
                         iterations=bm_cat["cat_ne"]),
    "rf" : RandomForestClassifier(n_estimators=bm_rf["rf_ne"],
                         max_depth=bm_rf["rf_md"],
                         max_features=bm_rf["rf_mf"], n_jobs=2),
    "ebm": ExplainableBoostingClassifier(
                         max_bins=bm_ebm["ebm_bins"],
                         max_interaction_bins=bm_ebm["ebm_inter"]),
    #"tab": TabNetClassifier(n_d=bm_tab["tabnet_n_d"],
                        # n_a=bm_tab["tabnet_n_d"],
                       #  n_shared=bm_tab["tabnet_n_d"],
                       #  n_steps=bm_tab["tabnet_n_steps"],
                       #  output_dim=N_CLASSES,
                       #  optimizer_fn=torch.optim.Adam,
                     #    optimizer_params=dict(lr=1e-3),
                         #device_name="cpu", verbose=0)
}
# nachdem X, y, w erzeugt wurden
dt_like = X.select_dtypes(include=["datetime", "datetimetz"]).columns
if len(dt_like):
    logging.info("⏱  drop datetime cols → %s", list(dt_like))
    X = X.drop(columns=dt_like)

# ---------- 4) Stacking + Kalibrierung --------------
w  = compute_sample_weight(class_weight={0:1.0, 1:2.5}, y=y)

cv = StratifiedKFold(n_splits=KFOLD, shuffle=False)

stack = StackingClassifier(
        estimators=list(models.items()),
        final_estimator=LogisticRegression(
            C=meta_C, max_iter=1000, multi_class="ovr",
            class_weight={0:1.0, 1:2.5}),
        stack_method="predict_proba", cv=cv, n_jobs=2)

stack.fit(X, y, sample_weight=w)

clf = CalibratedClassifierCV(stack, cv=5, method="isotonic")
clf.fit(X, y, sample_weight=w)

# ---------- 5) Threshold bestimmen ------------------
proba = clf.predict_proba(X)[:, 1]
fpr, tpr, thr = roc_curve(y, proba, pos_label=1)

target_recall = 0.58
idx           = np.argmin(np.abs(tpr - target_recall))
best_thr      = float(thr[idx])

logging.info("Gewählter SHORT-Schwellenwert: %.4f (Recall ≈ %.2f %%)",
             best_thr, tpr[idx]*100)

# ---------- 6) Speichern -----------------------------
joblib.dump({"model": clf, "thr_short": best_thr}, OUT_PATH, compress=3)
logging.info("✓ Modell + Schwelle gespeichert → %s", OUT_PATH)

# ---------- 7) Kurz-Report ---------------------------
pred = (proba >= best_thr).astype(int)          # Schwelle anwenden
log  = log_loss(y, proba)
f1   = f1_score(y, pred, average="macro")
acc  = accuracy_score(y, pred)
bal  = balanced_accuracy_score(y, pred)

logging.info("Hold-in LogLoss: %.4f | Acc: %.4f | BalAcc: %.4f | F1: %.4f",
             log, acc, bal, f1)
logging.info("\n%s", classification_report(y, pred))
logging.info("CM\n%s", confusion_matrix(y, pred, labels=[0,1]))
