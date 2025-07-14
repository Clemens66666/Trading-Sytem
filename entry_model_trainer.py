#!/usr/bin/env python3
# ===================================================================
#  Entry-Model-Trainer  •  full Optuna pipeline  (CHUNK-Version)
# ===================================================================
#  • Feature-Tuning  (Log-Loss ↓)
#  • Label-Tuning    (High/Low-Zones)
#  • Basis-Modelle   (7 Algo-Typen, Optuna)
#  • Meta-Tuning     (C der LogReg)
#  • RAM-schonend    durch Chunk-Processing (bis 10 Mio Zeilen)
# ===================================================================

# ────────────────────────── Standard-Libs ──────────────────────────
import os, time, warnings, logging, joblib, gc
from datetime import timedelta

# ─────────────────────────── 3rd-Party ─────────────────────────────
import numpy as np
import pandas as pd
import optuna

from pathlib  import Path
from typing   import Dict, List, Optional, Tuple, Union

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.metrics       import roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   HistGradientBoostingClassifier,
                                   StackingClassifier)
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, f1_score      #  ← NEU
# GANZ OBEN im Skript (bei den anderen Imports):
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score,
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier, CatBoostError   
from interpret.glassbox    import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from ngboost               import NGBClassifier

from optuna.samplers       import TPESampler
from optuna.pruners        import MedianPruner
from optuna.exceptions     import TrialPruned

import torch                                # <── NEU für TabNet-CPU
from pytorch_tabnet.tab_model import TabNetClassifier

from mlfinlab.cross_validation import PurgedKFold
from CustomPurgedKFold         import CustomPurgedKFold   # eigener Wrapper

# ── Funktionen des Long-Trends wiederverwenden ────────────────
from features_utils import make_hourly_bars          # 1-H-Bars


# ─────────────────────── Globale Konstanten ───────────────────────
# Pfade
DATA_RAW   = "E:/RawTickData3.txt"
LONG_MODEL = Path("E:/longtrend_ensemble_dl.pkl")
OUT_PATH   = "E:/entry_stack_fullopt1min.pkl"
LOG_FILE   = "E:/logs/entry_trainer.log"

# Resample-Regeln & Daten-Chunks
BAR_RULE_MIN   = "1T"
BAR_RULE_HOUR  = "1H"
KFOLD          = 6
CHUNK          = 15_000_000          # ≈ 250 MB CSV-Block
EPS            = 1e-6

# ───────────────── Label-Tuning (High/Low-Zones) ──────────────────
STUDY_LABEL_ZONES   = "label_zone8"
N_TRIALS_LABEL_ZONE = 5         # Anzahl Optuna-Trials
WINDOW_MIN          = 2          # Minimales Fenster in Minuten
WINDOW_MAX          = 100        # Maximales Fenster in Minuten


# Optuna
OPTUNA_DB      = "sqlite:///optuna.db"
STUDY_SENTIMENT   = "sent_chunk10"
STUDY_FEATURE     = "feat_chunk10"
STUDY_BASE_PREFIX = "base13_"
STUDY_META        = "meta_logreg12"

N_TRIALS_SENTIMENT = 50     # Sentiment-Fenster
N_TRIALS_FEATURE   = 30   # Technische Features
N_TRIALS_BASE      = 15     # pro Basis-Modell
N_TRIALS_META      = 25     # Meta-Regressor

# Labeling-Fenster (High/Low-Zones)
LABEL_WINDOW_MINUTES = 60

# ─── am Anfang (nach den anderen Konstanten) ──────────────────────
N_CLASSES = 2          # 1 = HIGH-Zone (Long), 2 = LOW-Zone (Short)


# Logging-Setup
os.makedirs(Path(LOG_FILE).parent, exist_ok=True)
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()]
)
warnings.filterwarnings("ignore")

# ══════════════════ Helper-Funktionen ══════════════════════════════
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "TimeStamp" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "TimeStamp"})
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    elif "Time" in df.columns:
        df = df.rename(columns={"Time": "TimeStamp"})
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    else:
        raise ValueError("Keine Zeitspalte gefunden.")
    return df

def standardize_tick_last(df: pd.DataFrame) -> pd.DataFrame:
    if "Tick_Last" in df.columns:
        return df
    for c in ["close", "Close", "price", "Price"]:
        if c in df.columns:
            df["Tick_Last"] = df[c]
            return df
    raise ValueError("Keine Preis-Spalte erkannt.")

def resample_to_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = ensure_timestamp(df).pipe(standardize_tick_last)
    df = df.set_index("TimeStamp").sort_index()
    ohlc = df["Tick_Last"].resample(rule).ohlc()
    vol  = df.get("Tick_Volume", pd.Series(1.0, index=df.index))\
             .resample(rule).sum()
    vwap = (df["Tick_Last"] * df.get("Tick_Volume", 1.0))\
             .resample(rule).sum() / (vol + EPS)
    bars = ohlc.join(vol.rename("Tick_Volume"))
    bars["vwap"]        = vwap
    bars["bar_range"]   = bars["high"] - bars["low"]
    bars["bar_return"]  = (bars["close"] - bars["open"]) / (bars["open"] + EPS)
    bars["Tick_Last"]   = bars["close"]
    return bars.reset_index()

# ───────────────────────── OFI-Berechnung ─────────────────────────
def compute_ofi_min(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    df = ensure_timestamp(raw).pipe(standardize_tick_last)
    df["Tick_Volume"] = df.get("Tick_Volume", 1.0)
    df = df.sort_values("TimeStamp")
    df["tick_sign"] = np.sign(df["Tick_Last"].diff()).fillna(0)
    df["ofi_raw"]   = df["tick_sign"] * df["Tick_Volume"]
    df["ofi_tick"]  = df["ofi_raw"].rolling(window, 1).sum() / \
                      (df["Tick_Volume"].rolling(window, 1).sum() + EPS)
    return (df.set_index("TimeStamp")["ofi_tick"]
              .resample(BAR_RULE_MIN).last().ffill()
              .rename("ofi_min").reset_index())

# ────────────────────── Sentiment & Features ──────────────────────
class MarketSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window=158, w1=.8, w2=.1, w3=.1):
        self.window, self.w1, self.w2, self.w3 = window, w1, w2, w3

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kw):
        df = ensure_timestamp(X).pipe(standardize_tick_last)
        df["Tick_Volume"] = df.get("Tick_Volume", 1.0)

        ret  = df["Tick_Last"].pct_change().fillna(0)
        mom  = df["Tick_Last"] - df["Tick_Last"].shift(self.window)
        vol  = df["Tick_Last"].rolling(self.window, 1).std().fillna(EPS)
        volC = df["Tick_Volume"] / \
               (df["Tick_Volume"].rolling(self.window, 1).mean() + EPS) - 1

        df["sentiment_score"] = (self.w1 * ret + self.w2 * mom + self.w3 * volC) / (vol + EPS)
        return df

class FeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, vs=5, vl=20, r1=1, r5=5,
                 rsi_w=14, mf=12, ms=26, msig=9):
        self.vs, self.vl = vs, vl
        self.r1, self.r5 = r1, r5
        self.rsi_w, self.mf, self.ms, self.msig = rsi_w, mf, ms, msig

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kw):
        df = X.copy()
        df["vol_short"] = df["Tick_Last"].rolling(self.vs).std().fillna(0)
        df["vol_long"]  = df["Tick_Last"].rolling(self.vl).std().fillna(0)
        df["ret_1"]     = df["Tick_Last"].pct_change(self.r1).fillna(0)
        df["ret_5"]     = df["Tick_Last"].pct_change(self.r5).fillna(0)

        delta = df["Tick_Last"].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        rs    = gain.rolling(self.rsi_w).mean() / \
                loss.rolling(self.rsi_w).mean().replace(0, np.nan)
        df["rsi"] = (100 - 100 / (1 + rs)).fillna(0)

        exp1 = df["Tick_Last"].ewm(span=self.mf, adjust=False).mean()
        exp2 = df["Tick_Last"].ewm(span=self.ms, adjust=False).mean()
        macd = exp1 - exp2
        df["macd"]        = macd
        df["macd_signal"] = macd.ewm(span=self.msig, adjust=False).mean()
        return df.fillna(0)


# ─────────────────── Long-Trend-Probability-Helpers ───────────────────
def compute_trend_probabilities(
        bases: List[BaseEstimator],
        meta:  BaseEstimator,
        feat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:

    chunksize       = 10_000_000
    base_p_list, hours_list = [], []
    feat_cols_clean = [c for c in feat_cols if c.lower() != "timestamp"]

    for chunk in pd.read_csv(DATA_RAW, chunksize=chunksize):
        # ── 1) Zeit & Preis harmonisieren ────────────────────────────────
        chunk = ensure_timestamp(chunk)
        if "TimeStamp" in chunk.columns:
            chunk = chunk.rename(columns={"TimeStamp": "timestamp"})
        elif "Time" in chunk.columns:
            chunk = chunk.rename(columns={"Time": "timestamp"})

        for col in ["Bid", "price", "Price", "Tick_Last", "close", "Close"]:
            if col in chunk.columns:
                chunk = chunk.rename(columns={col: "Bid"})
                break
        else:
            raise ValueError("❌ Keine Preis-Spalte gefunden.")

        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.set_index("timestamp")

        # ── 2) 1-h-Bars ----------------------------------------------------
        h = make_hourly_bars(chunk)            # Spalte timestamp ODER TimeStamp
        if "timestamp" in h.columns:           # vereinheitlichen!
            h = h.rename(columns={"timestamp": "TimeStamp"})
        h = h.sort_values("TimeStamp")

        # ── 3) Basis-Modelle ---------------------------------------------
        X_chunk = h[feat_cols_clean].fillna(0)
        base_p_chunk = np.stack(
            [m.predict_proba(X_chunk)[:, 1] for m in bases], axis=1)

        base_p_list.append(base_p_chunk)
        hours_list.append(h)

    # ── 4) Zusammenführen & Meta-Proba -----------------------------------
    hours = pd.concat(hours_list, ignore_index=True)     # hat garantiert TimeStamp
    base_p_all = np.concatenate(base_p_list, axis=0)

    p_long = base_p_all.mean(axis=1)
    X_meta = np.column_stack([1.0 - p_long, p_long])

    trend_long_ser = pd.Series(
        meta.predict_proba(X_meta)[:, 1],
        index=pd.to_datetime(hours["TimeStamp"]),
        name="trend_prob_long"
    )

    return hours, trend_long_ser

# ───────────────────── 10-Min-Bars + ATR ──────────────────────────
def prepare_10min_bars(hours_df, trend_long_series):
    # egal ob TimeStamp Spalte oder Index – wir sorgen für die Spalte
    if "TimeStamp" not in hours_df:
        if isinstance(hours_df.index, pd.DatetimeIndex):
            hours_df = hours_df.reset_index().rename(columns={"index": "TimeStamp"})
        else:
            raise ValueError("hours_df hat weder TimeStamp-Spalte noch DatetimeIndex.")

    hours_df = hours_df.loc[~hours_df["TimeStamp"].duplicated(keep="last")]

    mins = resample_to_bars(hours_df, BAR_RULE_MIN).set_index("TimeStamp")
    trend_long_series = trend_long_series[~trend_long_series.index.duplicated(keep="last")]

    mins["trend_prob_long"] = trend_long_series.reindex(mins.index, method="ffill").values
    mins["ATR"] = mins["close"].rolling(14).std().bfill()
    return mins.reset_index()

# ─────────────────────── Feature-Frame (Chunk) ────────────────────
def build_feature_df_chunks(
        sent_par: dict,
        feat_par: dict,
        chunksize: int = CHUNK
) -> pd.DataFrame:
    dfs = []

    vs      = feat_par.get("vs",   feat_par.get("vol_short", 20))
    vl      = feat_par.get("vl",   feat_par.get("vol_long", 60))
    r1      = feat_par.get("r1",   feat_par.get("ret1", 1))
    r5      = feat_par.get("r5",   feat_par.get("ret5", 5))
    rsi_w   = feat_par.get("rsi",  feat_par.get("rsi_w", 14))
    mf      = feat_par.get("mf",   12)
    ms      = feat_par.get("ms",   26)
    msig    = feat_par.get("sig",  feat_par.get("msig", 9))
    ofi_win = feat_par.get("ofi_window", feat_par.get("ofi", 120))

    for chunk in pd.read_csv(DATA_RAW, chunksize=chunksize):
        chunk = standardize_tick_last(ensure_timestamp(chunk))

        # 10-Min-Bars
        mins  = resample_to_bars(chunk, BAR_RULE_MIN)

        # OFI
        ofi   = compute_ofi_min(chunk, ofi_win)
        mins  = pd.merge_asof(mins.sort_values("TimeStamp"), ofi,
                              on="TimeStamp", direction="backward")

        # Sentiment (auf Stunden-Bars)
        mst   = MarketSentimentTransformer(
                    sent_par["long_window"],
                    sent_par.get("r1", .8),
                    sent_par.get("r2", .1),
                    sent_par.get("r3", .1))
        hours = resample_to_bars(chunk, BAR_RULE_HOUR)
        sent  = mst.transform(hours)[["TimeStamp", "sentiment_score"]]

        mins  = pd.merge_asof(mins.sort_values("TimeStamp"), sent,
                              on="TimeStamp", direction="backward")

        # Technische Features
        fe_aug = FeatureAugmenter(vs, vl, r1, r5, rsi_w, mf, ms, msig)
        dfs.append(fe_aug.transform(mins))

    return pd.concat(dfs, ignore_index=True).fillna(0)

# ════════════════  Optuna – Sentiment-Tuning  ═════════════════════
def tune_sentiment_chunks(
        n_trials: int = N_TRIALS_SENTIMENT,
        chunksize: int = 10_000_000,
        study_name: str = STUDY_SENTIMENT
) -> dict:
    # 1) alle 4-h-Bars sammeln
    hours_list = []
    for chunk in pd.read_csv(DATA_RAW, chunksize=chunksize):
        chunk = standardize_tick_last(ensure_timestamp(chunk))
        h_4h  = resample_to_bars(chunk, BAR_RULE_HOUR)
        hours_list.append(h_4h)
    hours = pd.concat(hours_list, ignore_index=True)

    # 2) Ziel-Variable: Next-Up (1 Tag Shift)
    hours = hours.sort_values("TimeStamp")
    fut   = hours[["TimeStamp", "Tick_Last"]].rename(
                columns={"TimeStamp": "TargetTime", "Tick_Last": "FuturePrice"})
    hours["TargetTime"] = hours["TimeStamp"] + timedelta(days=1)
    hours = pd.merge_asof(hours, fut, on="TargetTime", direction="forward")
    hours["NextUp"] = (hours["FuturePrice"] > hours["Tick_Last"]).astype(int)

    # 3) Objective
    def objective(trial):
        win = trial.suggest_int("long_window", 24, 600)
        mst    = MarketSentimentTransformer(window=win)
        feats  = mst.transform(hours)[["sentiment_score"]].fillna(0.0)
        target = hours["NextUp"].values

        cv     = TimeSeriesSplit(n_splits=KFOLD)
        aucs   = []
        for tr, te in cv.split(feats):
            y_te = target[te]
            if len(np.unique(y_te)) < 2:
                continue
            lr = LogisticRegression(solver="liblinear", C=1.0)
            lr.fit(feats.iloc[tr], target[tr])
            p  = lr.predict_proba(feats.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y_te, p))

        if not aucs:
            raise TrialPruned()
        return float(np.mean(aucs))

    st = optuna.create_study(direction="maximize",
                             study_name=study_name,
                             storage=OPTUNA_DB,
                             load_if_exists=True)
    st.optimize(objective, n_trials=n_trials)

    best = st.best_trial.params
    logging.info("Bestes Sentiment-Fenster: %s", best)
    best.update({"r1": .6, "r2": .3, "r3": .1})
    return best

# ════════════════  Optuna – Feature-Tuning  ═══════════════════════
def tune_features_chunks(sent_par, n_trials=N_TRIALS_FEATURE):
    CHUNK_LOCAL = 10_000_000

    # 1) gesamte 10-Min-Serie
    mins_all = []
    for chunk in pd.read_csv(DATA_RAW, chunksize=CHUNK_LOCAL):
        chunk = standardize_tick_last(ensure_timestamp(chunk))
        mins_all.append(resample_to_bars(chunk, BAR_RULE_MIN))
    mins_all = pd.concat(mins_all, ignore_index=True)

    y = (mins_all["close"].shift(-1) > mins_all["close"]).astype(int).iloc[:-1]
    samples_info = mins_all.iloc[:-1]["TimeStamp"]

    def objective(trial):
        p = dict(
            vol_short  = trial.suggest_int ("vs",   5,  40),
            vol_long   = trial.suggest_int ("vl",  20, 120),
            ret1       = trial.suggest_int ("r1",   1,  10),
            ret5       = trial.suggest_int ("r5",   5,  60),
            rsi_w      = trial.suggest_int ("rsi", 10,  60),
            mf         = trial.suggest_int ("mf",   5,  30),
            ms         = trial.suggest_int ("ms",  15,  60),
            msig       = trial.suggest_int ("sig",  5,  40),
            ofi_window = trial.suggest_int ("ofi", 20, 300),
        )
        feats = (build_feature_df_chunks(sent_par, p)
                   .iloc[:-1]
                   .drop(columns=["TimeStamp", "Tick_Last", "Tick_Volume"])
                   .fillna(0.0))

        pkf = CustomPurgedKFold(KFOLD, samples_info, pct_embargo=0.01)
        logloss = []
        for tr, te in pkf.split(feats):
            mdl = HistGradientBoostingClassifier(loss="log_loss")
            mdl.fit(feats.iloc[tr], y.iloc[tr])
            prob = mdl.predict_proba(feats.iloc[te])[:, 1]
            ll   = log_loss(y.iloc[te], prob, labels=[0,1])
            logloss.append(ll)
        return float(np.mean(logloss))

    st = optuna.create_study(direction="minimize",
                             study_name   = STUDY_FEATURE,
                             storage      = OPTUNA_DB,
                             load_if_exists = True)
    st.optimize(objective, n_trials=n_trials)

    logging.info("Bestes Feature-Set: %s", st.best_trial.params)
    return st.best_trial.params

# ═══════════════════  LABEL-HILFEN  ══════════════════════════════
def _ensure_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sucht nach einer Preis-Spalte und benennt sie in 'price' um.
    Akzeptiert: price | close | Close | Tick_Last | Last | Bid
    """
    if "price" in df.columns:
        return df

    for cand in ["close", "Close", "Tick_Last", "Last", "Bid"]:
        if cand in df.columns:
            return df.rename(columns={cand: "price"})
    raise KeyError(
        "bars_df enthält keine erkannte Preis-Spalte "
        "(price / close / Tick_Last / Last / Bid)."
    )


# ════════════════  Labeling  (High/Low-Zones)  ════════════════════
def label_high_low_df(bars_df: pd.DataFrame,
                      window_minutes: int = LABEL_WINDOW_MINUTES
) -> pd.DataFrame:
    """
    Markiert in jedem *window_minutes*-Fenster das lokale Hoch (HIGH_ZONE)
    und Tief (LOW_ZONE).
    Rückgabe-DF: timestamp | zone_type | price
    """
    df = _ensure_price_column(bars_df.copy())

    # evtl. ungültige Preise verwerfen
    df = df.dropna(subset=["price"])
    if df.empty:
        raise ValueError("bars_df enthält nur NaN-Preise.")

    # Zeitachse vereinheitlichen
    if "TimeStamp" in df.columns:
        df = df.rename(columns={"TimeStamp": "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["window_start"] = df["timestamp"].dt.floor(f"{window_minutes}T")

    labels = []
    for _, grp in df.groupby("window_start", sort=False):
        if grp["price"].notna().sum() == 0:
            continue
        idx_high = grp["price"].idxmax()
        idx_low  = grp["price"].idxmin()

        labels.append({
            "timestamp": df.loc[idx_high, "timestamp"],
            "zone_type": "HIGH_ZONE",
            "price":     df.loc[idx_high, "price"]
        })
        labels.append({
            "timestamp": df.loc[idx_low,  "timestamp"],
            "zone_type": "LOW_ZONE",
            "price":     df.loc[idx_low,  "price"]
        })

    return pd.DataFrame(labels)


# ═══════════ Optuna – Label-Fenster (High/Low-Zones) ════════════
def tune_zone_window(
        bars_df: pd.DataFrame,
        proc_feat_df: pd.DataFrame,
        n_trials: int = N_TRIALS_LABEL_ZONE,
        study_name: str = STUDY_LABEL_ZONES
) -> int:
    """
    Optimiert die Fensterlänge (in Minuten) für HIGH/LOW-Zonen-Labels.
    • Ziel-Metrik: 1 − AUC  (=> minimieren)
    • Pruning:
        – zu wenige Samples  (< 2 Klassen oder < 100 High-Labels)
        – High-Anteil < 30 %
    """

    # ───────────────────────── Objective ─────────────────────────
    def objective(trial: optuna.trial.Trial) -> float:
        win = trial.suggest_int("window_minutes", WINDOW_MIN, WINDOW_MAX)

        # ── Labels anlegen ───────────────────────────────────────
        labels_df = label_high_low_df(bars_df, window_minutes=win)
        df = proc_feat_df.merge(
            labels_df.rename(columns={"timestamp": "TimeStamp"}),
            on="TimeStamp", how="left"
        ).dropna(subset=["zone_type"])

        # 0 = HIGH, 1 = LOW
        y = df["zone_type"].map({"HIGH_ZONE": 0, "LOW_ZONE": 1}).astype(int)

        # ── Pruning ─────────────────────────────────────────────
        if y.nunique() < 2 or (y == 0).sum() < 100:
            raise optuna.TrialPruned()
        if (y == 0).mean() < 0.30:
            raise optuna.TrialPruned()

        # ── Feature-Matrix ──────────────────────────────────────
        X = (df.drop(columns=["zone_type", "Tick_Last", "Tick_Volume"], errors="ignore")
               .select_dtypes(exclude=[np.datetime64])
               .fillna(0.0))

        # ── CV mit Purged K-Fold ────────────────────────────────
        pkf = CustomPurgedKFold(
            n_splits=KFOLD,
            samples_info_sets=pd.Series(X.index),
            pct_embargo=0.01
        )

        aucs = []
        for tr_idx, te_idx in pkf.split(X):
            mdl = HistGradientBoostingClassifier(loss="log_loss")
            mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = mdl.predict_proba(X.iloc[te_idx])          # shape (n, 2)
            # AUC nur auf HIGH-Spalte (proba[:, 0])
            aucs.append(
                roc_auc_score((y.iloc[te_idx] == 0).astype(int), proba[:, 0])
            )

        return 1.0 - float(np.mean(aucs))  # minimieren

    # ─────────────────────── Study-Setup ────────────────────────
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    best_win = study.best_trial.params["window_minutes"]
    logging.info("🔖 Bestes Label-Fenster: %d Minuten", best_win)
    return best_win

# ═════════════════ Optuna – Basis-Modelle ═════════════════════════
ALL_MODELS = {
    "xgb":    XGBClassifier,
    "lgb":    LGBMClassifier,
    "cat":    CatBoostClassifier,
    "rf":     RandomForestClassifier,
    "ebm":    ExplainableBoostingClassifier,
    "tabnet": TabNetClassifier,
}

# ═══════════════ Optuna – Basis-Modelle (build_and_tune_base_models) ═══════════════
# ═══════════════ Optuna – Basis-Modelle (build_and_tune_base_models) ═══════════════
def build_and_tune_base_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_TRIALS_BASE
) -> Dict[str, BaseEstimator]:
    """
    • Hyper-Parameter-Tuning aller Basis-Modelle
    • Datetime-Spalten werden entfernt
    • Folds ohne beide Klassen werden übersprungen
    • Pruning, falls kein einziger gültiger Fold übrig bleibt
    """
    # ───────── Datetime-Spalten droppen ───────────────────────────
    dt_cols = [c for c, d in X.dtypes.items() if np.issubdtype(d, np.datetime64)]
    if dt_cols:
        logging.info("Dropping datetime cols before training: %s", dt_cols)
        X = X.drop(columns=dt_cols)

    n_feat = X.shape[1]
    models: Dict[str, BaseEstimator] = {}

    # ───────── Loop über alle Algorithmen ─────────────────────────
    for name, ModelClass in ALL_MODELS.items():

        # ─── Optuna-Objective ─────────────────────────────────────
        def objective(trial: optuna.trial.Trial) -> float:

            # — Hyper-Parameter — ---------------------------------
            if name == "xgb":
                model = XGBClassifier(
                    objective       = "multi:softprob",
                    num_class       = N_CLASSES,
                    learning_rate   = trial.suggest_float("xgb_lr", 1e-3, 1e-1, log=True),
                    max_depth       = trial.suggest_int  ("xgb_md", 3, 12),
                    n_estimators    = trial.suggest_int  ("xgb_ne", 50, 500),
                    subsample       = trial.suggest_float("xgb_ss", .3, 1.0),
                    n_jobs=1
                )
                
            elif name == "lgb":
                model = LGBMClassifier(
                    objective       = "multiclass",
                    num_class       = N_CLASSES,
                    learning_rate   = trial.suggest_float("lgb_lr", 1e-3, 1e-1, log=True),
                    max_depth       = trial.suggest_int  ("lgb_md", 3, 12),
                    n_estimators    = trial.suggest_int  ("lgb_ne", 50, 300),
                    subsample       = trial.suggest_float("lgb_ss", .3, 1.0),
                    n_jobs=1)

            elif name == "cat":
                model = CatBoostClassifier(
                    loss_function = "MultiClass",
                    classes_count = N_CLASSES,
                    learning_rate = trial.suggest_float("cat_lr", 1e-3, 1e-1, log=True),
                    depth         = trial.suggest_int  ("cat_md", 3, 12),
                    iterations    = trial.suggest_int  ("cat_ne", 50, 300),
                    verbose=0
                )

            elif name == "rf":
                model = RandomForestClassifier(
                    n_estimators = trial.suggest_int  ("rf_ne", 50, 500),
                    max_depth    = trial.suggest_int  ("rf_md", 3, 20),
                    max_features = trial.suggest_float("rf_mf", .3, 1.0),
                    n_jobs=1)

            elif name == "ebm":
                model = ExplainableBoostingClassifier(
                    max_bins            = trial.suggest_int("ebm_bins",  3, 50),
                    max_interaction_bins= trial.suggest_int("ebm_inter", 3, 10))

            elif name == "tabnet":
                nd = trial.suggest_int("tabnet_n_d", 8, min(32, n_feat))
                ns = trial.suggest_int("tabnet_n_steps", 3, 7)
                model = TabNetClassifier(
                    n_d=nd, n_a=nd, n_shared=nd, n_steps=ns,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=1e-3),
                    output_dim=N_CLASSES,
                    device_name="cpu",  verbose=0
                )

            else:
                model = ModelClass()

            pkf = CustomPurgedKFold(
                n_splits          = KFOLD,
                samples_info_sets = pd.Series(X.index),
                pct_embargo       = 0.0)

            fold_losses = []
            for tr_idx, te_idx in pkf.split(X):
                y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

                if y_tr.nunique() < 2 or y_te.nunique() < 2:
                    continue                                 # Skip-Fold

                try:
                    if name == "tabnet":                     # TabNet braucht np.float32
                        X_tr = X.iloc[tr_idx].values.astype(np.float32)
                        X_te = X.iloc[te_idx].values.astype(np.float32)

                        model.fit(
                            X_tr, y_tr.values,
                            eval_set=[(X_te, y_te.values)],
                            eval_metric=["logloss"],
                            max_epochs=50, patience=5, batch_size=1024
                        )
                        preds = model.predict_proba(X_te)          # (n,2)
                    else:
                        model.fit(X.iloc[tr_idx], y_tr)
                        preds = model.predict_proba(X.iloc[te_idx])

                except (RuntimeError, np.linalg.LinAlgError):
                    raise optuna.TrialPruned()

                loss = log_loss(y_te, preds, labels=[0,1])
                fold_losses.append(loss)

            if not fold_losses:
                raise optuna.TrialPruned()

            return float(np.mean(fold_losses))            # minimieren

        # — Optuna-Study — ----------------------------------------
        study = optuna.create_study(
            study_name     = f"{STUDY_BASE_PREFIX}{name}",
            storage        = OPTUNA_DB,
            load_if_exists = True,
            direction      = "minimize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial.params
        logging.info("📊 %s best params: %s", name.upper(), best)

        # — Final-Fit (Best-Params) — ------------------------------
        if name == "tabnet":
            m = TabNetClassifier(
                    n_d            = best["tabnet_n_d"],
                    n_a            = best["tabnet_n_d"],
                    n_shared       = best["tabnet_n_d"],
                    n_steps        = best["tabnet_n_steps"],
                    output_dim     = N_CLASSES, 
                    optimizer_fn   = torch.optim.Adam,
                    optimizer_params=dict(lr=1e-3),
                    device_name    = "cpu",
                    verbose        = 0
                ).fit(
                    X.values.astype(np.float32),
                    y.values,
                    max_epochs = 50,
                    patience   = 7,
                    batch_size = 4096
                )

        elif name == "xgb":
            m = XGBClassifier(
             objective       = "multi:softprob",
             num_class       = N_CLASSES,
             learning_rate   = best["xgb_lr"],
              max_depth       = best["xgb_md"],
              n_estimators    = best["xgb_ne"],
             subsample       = best["xgb_ss"],
              n_jobs          = 1
        ).fit(X, y)

        elif name == "lgb":
            m = LGBMClassifier(
             objective     = "multiclass",
             num_class     = N_CLASSES,
              learning_rate = best["lgb_lr"],
              max_depth     = best["lgb_md"],
              n_estimators  = best["lgb_ne"],
              subsample     = best["lgb_ss"],
             n_jobs        = 1
        ).fit(X, y)

        elif name == "cat":
            m = CatBoostClassifier(
              loss_function = "MultiClass",
             classes_count = N_CLASSES,
             learning_rate = best["cat_lr"],
             depth         = best["cat_md"],
              iterations    = best["cat_ne"],
             verbose       = 0
        ).fit(X, y)

        elif name == "rf":
            m = RandomForestClassifier(
                    n_estimators   = best["rf_ne"],
                    max_depth      = best["rf_md"],
                    max_features   = best["rf_mf"],
                    n_jobs=1).fit(X, y)

        elif name == "ebm":
            m = ExplainableBoostingClassifier(
                    max_bins              = best["ebm_bins"],
                    max_interaction_bins  = best["ebm_inter"]).fit(X, y)

        else:  # Fallback (sollte nicht eintreten)
            m = ModelClass().fit(X, y)

        models[name] = m

    return models


# ═══════════════ Out-of-Fold-Prognosen (get_oof_predictions) ═══════════════
def get_oof_predictions(
    models,                     # Dict ODER List[(name, mdl)]
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:

    # -------- List → Dict ----------------------------------------
    if isinstance(models, list):
        models = dict(models)

    # -------- Datetime-Spalten entfernen -------------------------
    dt_cols = [c for c, d in X.dtypes.items() if np.issubdtype(d, np.datetime64)]
    if dt_cols:
        logging.info("OOF: dropping datetime cols %s", dt_cols)
        X = X.drop(columns=dt_cols)

    valid_models = {n: m for n, m in models.items() if m is not None}
    if not valid_models:
        raise ValueError("Es existiert kein trainiertes Basis-Modell!")

    oof = np.full(
        (len(X), len(valid_models) * N_CLASSES),
        0.5, dtype=float
    )

    pkf = CustomPurgedKFold(
            n_splits          = KFOLD,
            samples_info_sets = pd.Series(X.index),
            pct_embargo       = 0.01
          )

    for k, (name, mdl) in enumerate(valid_models.items()):
        for tr_idx, te_idx in pkf.split(X):
            try:
                if isinstance(mdl, TabNetClassifier):
                    mdl.fit(
                        X.iloc[tr_idx].values.astype(np.float32), y.iloc[tr_idx].values,
                        eval_set=[(
                            X.iloc[te_idx].values.astype(np.float32),
                            y.iloc[te_idx].values)],
                        eval_metric=["logloss"],
                        max_epochs=50, patience=5, batch_size=1024
                    )
                    preds = mdl.predict_proba(
                        X.iloc[te_idx].values.astype(np.float32))
                else:
                    mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    preds = mdl.predict_proba(X.iloc[te_idx])

                oof[te_idx, k*N_CLASSES:(k+1)*N_CLASSES] = preds

            except (RuntimeError, np.linalg.LinAlgError, CatBoostError) as err:
                logging.warning("⚠️  %s fold skipped (%s)", name.upper(), err)
                continue

    cols = [f"{n}_p{c}" for n in valid_models for c in range(N_CLASSES)]
    return pd.DataFrame(oof, index=X.index, columns=cols)


# ───────────────────────── Meta-LogReg-Tuning ────────────────────────────
def tune_meta_logreg(
    X: pd.DataFrame,   # OOF-Proba-Matrix (multiclass)
    y: pd.Series,
    n_trials: int = 25
) -> float:

    def obj(trial):
        C_val = trial.suggest_float("C", 1e-3, 10, log=True)
        lr    = LogisticRegression(C=C_val, multi_class="ovr", max_iter=1000)

        pkf = CustomPurgedKFold(
            n_splits          = KFOLD,
            samples_info_sets = pd.Series(X.index),
            pct_embargo       = 0.01
        )

        losses = []
        for tr_idx, te_idx in pkf.split(X):
            lr.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = lr.predict_proba(X.iloc[te_idx])
            losses.append(log_loss(y.iloc[te_idx], p, labels=[0,1]))

        return np.mean(losses)          # minimieren

    st = optuna.create_study(
        study_name = STUDY_META,
        storage       = OPTUNA_DB,
        direction     = "minimize",
        load_if_exists=True
    )
    st.optimize(obj, n_trials=n_trials)

    return st.best_params["C"]

# ─────────────────────────────  MAIN  ──────────────────────────────
def main() -> None:
    t0 = time.time()
    logging.info("Starte Entry-Trainer …")

    # 0) Long-Trend-Ensemble laden -----------------------------------------
    ens       = joblib.load(LONG_MODEL)
    bases     = ens["base_models"]
    meta      = ens["meta"]
    feat_cols = [c for c in ens["feat_cols"] if c.lower() != "timestamp"]

    # 1) Stunden-Bars + Trend-Probas (streamend) ---------------------------
    hours, trend_long = compute_trend_probabilities(bases, meta, feat_cols)

    # 1b) Trend-Cache ------------------------------------------------------
    global TREND_SER
    TREND_SER = trend_long

    # 2) 10-Min-Bars vorbereiten ------------------------------------------
    mins = prepare_10min_bars(hours, trend_long)

    # 3) Sentiment-Parameter (Optuna) -------------------------------------
    sent_par = tune_sentiment_chunks(n_trials=300)

    # 4) Feature-Parameter (Optuna) ---------------------------------------
    feat_par = tune_features_chunks(sent_par, n_trials=100)

    # 5) Minuten-Features bauen ------------------------------------------
    proc = build_feature_df_chunks(sent_par, feat_par)
    proc = proc.merge(
        mins[["TimeStamp", "trend_prob_long", "ATR"]],
        on="TimeStamp", how="left"
    )

    # ------------------------------------------------------------------
    # 6) Label-Fenster (Optuna) –  Multiclass (0=HIGH, 1=LOW)
    # ------------------------------------------------------------------
    best_win = tune_zone_window(resample_to_bars(proc, BAR_RULE_MIN), proc)

    labels_df = label_high_low_df(
      resample_to_bars(proc, BAR_RULE_MIN),
      window_minutes=best_win
    )

    labels_simple = (
        labels_df
         .assign(entry_label = np.where(labels_df["zone_type"] == "HIGH_ZONE", 0, 1))
         .rename(columns={"timestamp": "TimeStamp"})
          [["TimeStamp", "entry_label"]]
    )

    proc = proc.merge(labels_simple, on="TimeStamp", how="inner")
    proc["entry_label"] = proc["entry_label"].astype(int)


    logging.info("Label-Verteilung (1=High, 2=Low): %s",
             proc["entry_label"].value_counts().to_dict())


    # 7) Train / Hold-out-Split -------------------------------------------
    cutoff   = pd.Timestamp("2025-01-02")
    train_df = proc[proc["TimeStamp"] <  cutoff].copy()
    hold_df  = proc[proc["TimeStamp"] >= cutoff].copy()

    drop_cols = ["entry_label", "Tick_Last", "Tick_Volume"]
    X_tr = train_df.drop(columns=[c for c in drop_cols if c in train_df]).fillna(0)
    y_tr = train_df["entry_label"].astype(int)
    X_h  = hold_df.drop(columns=[c for c in drop_cols if c in hold_df]).fillna(0)
    y_h  = hold_df["entry_label"].astype(int)

    # ------------------------------------------------
    # 8) Basis-Modelle (Optuna) -----------------------
    tuned_models = build_and_tune_base_models(X_tr, y_tr, n_trials=30)

    # -----------------------------------------------------------------------       
    # 9) Meta-Regressor-Tuning + Stacking
    # ---------------------------------------------------------------------
    from utils_stacking import filter_valid_estimators          #  ← Utility

    # 9a) Valide Basis-Modelle identifizieren
    valid_models = dict(filter_valid_estimators(tuned_models))
    if len(valid_models) < 2:
        raise ValueError(
            f"Für den Stacker sind mindestens zwei valide Modelle nötig "
            f"(gefunden: {len(valid_models)})."
        )

    # 9b) OOF-Predictions für LogReg-Optuna
    oof    = get_oof_predictions(valid_models, X_tr, y_tr)
    best_C = tune_meta_logreg(oof, y_tr, n_trials=N_TRIALS_META)

    # 9c) Datumsspalten droppen
    dt_cols = [c for c, d in X_tr.dtypes.items() if np.issubdtype(d, np.datetime64)]
    if dt_cols:
        logging.info("Stacking: dropping datetime cols %s", dt_cols)
    X_tr_ndt = X_tr.drop(columns=dt_cols)
    X_h_ndt  = X_h .drop(columns=dt_cols)

    # ------------- 9d) Stacking-Classifier ---------------------------
    final_estimator = LogisticRegression(
        C           = best_C,
        solver      = "lbfgs",
        max_iter    = 1000,
        multi_class = "ovr",          # ← MULTICLASS
        n_jobs      = 2
    )

    stacker = StackingClassifier(
        estimators      = [(n, m) for n, m in valid_models.items()],
        final_estimator = final_estimator,
        stack_method    = "predict_proba",
        cv              = 5,
        passthrough     = False,
        n_jobs          = 2,
    )

    logging.info("🔗 Starte Stacking-Fit mit %d Basis-Modellen", len(valid_models))
    stacker.fit(X_tr_ndt, y_tr)
    logging.info("✅ Stacking-Fit abgeschlossen.")

    # 9e) Kalibrieren
    clf = CalibratedClassifierCV(stacker, method="sigmoid", cv=5).fit(X_tr_ndt, y_tr)
    joblib.dump(clf, OUT_PATH, compress=3)
    logging.info("✅ Modell gespeichert: %s", OUT_PATH)

    # -----------------------------------------------------------------
    # 10)  Validierung – “wahrscheinlichstes Label”
    # -----------------------------------------------------------------
    val_mask   = train_df["TimeStamp"] >= train_df["TimeStamp"].quantile(0.80)
    X_val      = X_tr_ndt[val_mask]
    y_val_true = y_tr[val_mask]

    y_val_pred = clf.predict(X_val)
    f1_val     = f1_score(y_val_true, y_val_pred, average="macro")
    logging.info("F1 (Val, macro): %.4f", f1_val)

    # -----------------------------------------------------------------
    # 11)  Hold-out
    # -----------------------------------------------------------------
    y_h_pred = clf.predict(X_h_ndt)

    metrics = {
        "Log-Loss"          : log_loss(y_h, clf.predict_proba(X_h_ndt), labels=[0, 1]),
        "Accuracy"          : accuracy_score(y_h, y_h_pred),
        "Balanced-Accuracy" : balanced_accuracy_score(y_h, y_h_pred),
        "F1 (macro)"        : f1_score(y_h, y_h_pred, average="macro"),
    }

    logging.info("── Hold-out-Metriken ──")
    for k, v in metrics.items():
       logging.info("%-17s %.4f", k + ":", v)

    cm = confusion_matrix(y_h, y_h_pred, labels=[0, 1])
    logging.info("Confusion-Matrix\n%s", cm)
    logging.info("\n%s", classification_report(y_h, y_h_pred, digits=4))

# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
