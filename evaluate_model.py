# ───────── Imports & Config ─────────
import os
import logging
import warnings
import time
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# CUDA‑Cache umleiten
os.environ["BOOST_COMPUTE_CACHE_DIR"] = r"C:\temp\boost_compute_cache"
os.makedirs(os.environ["BOOST_COMPUTE_CACHE_DIR"], exist_ok=True)

warnings.filterwarnings("ignore", module="lightgbm")

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from imblearn.over_sampling import SMOTE
import shap, joblib

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.pipeline        import Pipeline
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ───────── GLOBALS ─────────
DATA_RAW       = "E:/RawTickData2.txt"
MODEL_OUT      = "E:/trained_market_sentiment_model_pilot.pkl"
LOG_FILE       = "E:/logs/trainer_pilot.log"

N_SENT_TRIALS  = 200     # Pilot‑Run
N_LABEL_TRIALS = 300
KFOLD          = 10
SAMPLE_ROWS    = 1_000_000

# ───────── Logging ─────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ───────── Helpers ─────────
def ensure_timestamp(df):
    if "TimeStamp" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    elif "Time" in df.columns:
        df = df.rename(columns={"Time":"TimeStamp"})
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    else:
        raise ValueError("Keine Zeitspalte!")
    return df

def standardize_tick_last(df):
    for c in df.columns:
        if c.lower() == "tick_last" and c != "Tick_Last":
            df = df.rename(columns={c: "Tick_Last"})
    if "Tick_Last" not in df.columns:
        price_cols = [c for c in df.columns if "price" in c.lower()]
        df["Tick_Last"] = df[price_cols[0]]
    return df

# ───────── CUSUM‑Filter & Triple‑Barrier Labeling ─────────
def cusum_filter_dynamic(series, thr_series):
    t_events, s_pos, s_neg = [], 0.0, 0.0
    for i in range(1, len(series)):
        diff = series.iloc[i] - series.iloc[i-1]
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        thr = thr_series.iloc[i]
        if s_pos > thr or s_neg < -thr:
            t_events.append(series.index[i])
            s_pos, s_neg = 0.0, 0.0
    return t_events

def create_target(
    df,
    vol_window: int,
    barrier_multiplier: float,
    cusum_multiplier: float,
    horizon_seconds: int,
    min_event_gap_seconds: int = 60
) -> pd.DataFrame:
    ts = ensure_timestamp(df.copy()).sort_values("TimeStamp").set_index("TimeStamp")
    ts = standardize_tick_last(ts)
    ts["returns"]    = ts["Tick_Last"].pct_change()
    ts["volatility"] = ts["returns"].ewm(span=vol_window, adjust=False).std()
    local_thr = cusum_multiplier * ts["volatility"].rolling(vol_window, min_periods=1).mean()
    events    = cusum_filter_dynamic(ts["Tick_Last"], local_thr)

    rec, last_event = [], None
    for t in events:
        if last_event and (t - last_event).total_seconds() < min_event_gap_seconds:
            continue
        last_event = t
        p0, v0   = ts.at[t, "Tick_Last"], ts.at[t, "volatility"]
        barrier  = barrier_multiplier * v0 * p0
        end_time = t + pd.Timedelta(seconds=horizon_seconds)
        window = ts.loc[t:end_time]
        label = (
            1  if (window["Tick_Last"] >= p0 + barrier).any() else
           -1  if (window["Tick_Last"] <= p0 - barrier).any() else
            0
        )
        if label != 0:
            rec.append({"TimeStamp": t, "target": label, "price0": p0})

    df_ev = pd.DataFrame(rec)
    df_full = (
        df_ev
        .merge(ts.reset_index().drop(columns=["Tick_Last"]), on="TimeStamp", how="left")
        .rename(columns={"price0": "Tick_Last"})
    )
    return df_full

# ───────── Sentiment Transformer ─────────
class MarketSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window=50, w1=.3, w2=.5, w3=.2, eps=1e-6):
        self.window, self.w1, self.w2, self.w3, self.eps = window, w1, w2, w3, eps
    def fit(self, X, y=None): return self
    def transform(self, X, **kw):
        df = ensure_timestamp(X.copy())
        df = standardize_tick_last(df)
        df["Tick_Volume"] = df.get("Tick_Volume", 1.0)
        roll = df["Tick_Volume"].rolling(self.window, 1).mean()
        volC = df["Tick_Volume"]/(roll + self.eps) - 1
        ret  = df["Tick_Last"].pct_change().fillna(0)
        mom  = (df["Tick_Last"] - df["Tick_Last"].shift(self.window)).fillna(0)
        vol  = df["Tick_Last"].rolling(self.window, 1).std().fillna(self.eps)
        raw  = (self.w1*ret + self.w2*mom + self.w3*volC) / (vol + self.eps)
        df["sentiment_score"] = np.tanh(raw.astype(np.float32))
        return df

# ───────── Feature Augmentation ─────────
class FeatureAugmenter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, **kw):
        df = X.copy()
        df["vol_5"]   = df["Tick_Last"].rolling(5).std().fillna(0)
        df["vol_20"]  = df["Tick_Last"].rolling(20).std().fillna(0)
        df["ret_1"]   = df["Tick_Last"].pct_change().fillna(0)
        df["ret_5"]   = df["Tick_Last"].pct_change(5).fillna(0)
        df["hour"]    = df["TimeStamp"].dt.hour
        df["minute"]  = df["TimeStamp"].dt.minute
        return df

# ───────── Dynamic Aggregator ─────────
class DynAgg(BaseEstimator, TransformerMixin):
    def __init__(self, thr="2023-01-01", freq_recent="5min", freq_old="4h"):
        self.thr = pd.to_datetime(thr)
        self.fr, self.fo = freq_recent, freq_old
    def fit(self, X, y=None): return self
    def transform(self, X, **kw):
        df = ensure_timestamp(X.copy()).set_index("TimeStamp")
        old, rc = df[df.index < self.thr], df[df.index >= self.thr]
        agg = {c: "mean" for c in df.select_dtypes("number").columns if c != "Tick_Last"}
        agg["Tick_Last"] = "last"
        out = pd.concat([
            old.resample(self.fo).agg(agg),
            rc.resample(self.fr).agg(agg)
        ])
        return out.dropna().reset_index()

# ───────── Pilot‑Multi‑Scale Sentiment‑Tuning ─────────
def tune_sentiment(raw, n_trials):
    df_orig = ensure_timestamp(raw.copy()).sort_values("TimeStamp")
    df_orig = standardize_tick_last(df_orig)
    if len(df_orig) > SAMPLE_ROWS:
        df_orig = df_orig.sample(SAMPLE_ROWS, random_state=42)

    def objective(trial):
        # Zeithorizont
        horizon_days = trial.suggest_int("horizon_days", 10, 25)

        # Future‑Merge
        df = df_orig.copy()
        df["TargetTime"] = df["TimeStamp"] + pd.Timedelta(days=horizon_days)
        df = df.sort_values("TargetTime")
        future = (
            df[["TargetTime","Tick_Last"]]
            .rename(columns={"Tick_Last":"FuturePrice"})
            .sort_values("TargetTime")
        )
        df = pd.merge_asof(df, future, on="TargetTime", direction="forward")
        df["NextRet"]  = (df["FuturePrice"] - df["Tick_Last"]) / df["Tick_Last"]
        df["NextSign"] = df["NextRet"].gt(0).astype(int)

        # Pilot‑Suchraum
        short_win = trial.suggest_int("short_window", 650, 950)
        long_win  = trial.suggest_int("long_window", 600, 900)

        r1 = trial.suggest_float("w1_raw", 1.2, 2.0)
        r2 = trial.suggest_float("w2_raw", 0.0, 0.3)
        r3 = trial.suggest_float("w3_raw", 0.5, 1.2)
        s  = r1 + r2 + r3
        w1, w2, w3 = r1/s, r2/s, r3/s

        thr = trial.suggest_float("thr", 0.0, 0.1)

        # Scores
        ms = MarketSentimentTransformer(window=short_win, w1=w1, w2=w2, w3=w3)
        ml = MarketSentimentTransformer(window=long_win,  w1=w1, w2=w2, w3=w3)
        df_s = ms.transform(df.copy())[["TimeStamp","sentiment_score"]].sort_values("TimeStamp")
        df_l = ml.transform(df.copy())[["TimeStamp","sentiment_score"]].rename(
            columns={"sentiment_score":"sentiment_score_long"}
        ).sort_values("TimeStamp")
        dft = pd.merge_asof(df_s, df_l, on="TimeStamp", direction="nearest")

        # Zusatz‑Features mergen
        feat_df = (
            FeatureAugmenter()
            .transform(df.copy())
            [["TimeStamp","vol_5","vol_20","ret_1","ret_5","hour","minute"]]
            .sort_values("TimeStamp")
        )
        dft = pd.merge_asof(dft, feat_df, on="TimeStamp", direction="nearest")

        X = dft[[
            "sentiment_score","sentiment_score_long",
            "vol_5","vol_20","ret_1","ret_5","hour","minute"
        ]].values
        y = df["NextSign"].values

        calib = LogisticRegression(solver="liblinear").fit(X, y)
        prob  = calib.predict_proba(X)[:,1]
        preds = (prob > thr).astype(int)
        if preds.sum() < len(preds) * 0.005:
            return -np.inf, -np.inf, -np.inf

        roc = roc_auc_score(y, prob)
        ap  = average_precision_score(y, prob)
        pnl = (prob * df["NextRet"].values).sum()
        return roc, ap, pnl

    study = optuna.create_study(
        directions=["maximize"]*3,
        sampler=TPESampler(seed=42),
        pruner=SuccessiveHalvingPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = sorted(
        study.best_trials,
        key=lambda t: (t.values[2], t.values[0], t.values[1]),
        reverse=True
    )[0].params
    total = best["w1_raw"] + best["w2_raw"] + best["w3_raw"]
    out = {
        "w1":           best["w1_raw"]/total,
        "w2":           best["w2_raw"]/total,
        "w3":           best["w3_raw"]/total,
        "short_window": best["short_window"],
        "long_window":  best["long_window"],
        "horizon_days": best["horizon_days"],
        "thr":          best["thr"]
    }
    logging.info("Beste Sentiment‑Params: %s", out)
    return out

# ───────── Label‑Tuning ─────────
def tune_labeling(proc, n_trials):
    df = proc.copy()
    if len(df) > SAMPLE_ROWS:
        df = df.sample(SAMPLE_ROWS, random_state=43)

    def obj(trial):
        vol_w  = trial.suggest_int("vol_window",50,550)
        bar_m  = trial.suggest_float("barrier_multiplier",0.5,5.0)
        cus_m  = trial.suggest_float("cusum_multiplier",0.1,1.0)
        hor_s  = trial.suggest_int("horizon_seconds",60,3600)

        lab = create_target(
            df.copy(),
            vol_window=vol_w,
            barrier_multiplier=bar_m,
            cusum_multiplier=cus_m,
            horizon_seconds=hor_s
        )
        if lab["target"].nunique() < 2:
            return 0.0, 0.0, -np.inf

        feats = ["sentiment_score","sentiment_score_long"]
        X, y = lab[feats], lab["target"]
        Xb, yb = SMOTE(random_state=42).fit_resample(X, y)
        yb_enc = (yb>0).astype(int)
        if len(np.unique(yb_enc))<2:
            return 0.0, 0.0, -np.inf

        roc = np.nanmean(cross_val_score(
            LogisticRegression(max_iter=1000),
            Xb, yb_enc,
            cv=TimeSeriesSplit(KFOLD),
            scoring="roc_auc", n_jobs=1
        ))
        bal = 1 - abs(y.value_counts(normalize=True).diff().iloc[-1])
        lab["next_ret"] = lab["Tick_Last"].pct_change().shift(-1).fillna(0)
        pnl = (lab["target"]*lab["next_ret"]).sum()
        return roc, bal, pnl

    study = optuna.create_study(
        directions=["maximize"]*3,
        sampler=TPESampler(seed=42),
        pruner=SuccessiveHalvingPruner()
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    best = sorted(
        study.best_trials,
        key=lambda t: (t.values[2], t.values[0], t.values[1]),
        reverse=True
    )[0].params
    logging.info("Beste Label‑Params: %s", best)
    return best

# ───────── Feature‑Engineering ─────────
def build_feature_df(raw, sp):
    spc = {k:v for k,v in sp.items() if k in ("w1","w2","w3")}
    sw = sp["short_window"]
    lw = sp["long_window"]

    short = Pipeline([
        ("sent", MarketSentimentTransformer(window=sw, **spc)),
        ("feat", FeatureAugmenter()),
        ("agg",  DynAgg(freq_recent="5min", freq_old="4h"))
    ]).fit_transform(raw)

    long = Pipeline([
        ("sent", MarketSentimentTransformer(window=lw, **spc)),
        ("feat", FeatureAugmenter()),
        ("agg",  DynAgg(freq_recent="4h",  freq_old="4h"))
    ]).fit_transform(raw).rename(columns={"sentiment_score":"sentiment_score_long"})

    return pd.merge_asof(
        short.sort_values("TimeStamp"),
        long[[
            "TimeStamp","sentiment_score_long",
            "vol_5","vol_20","ret_1","ret_5","hour","minute"
        ]].sort_values("TimeStamp"),
        on="TimeStamp", direction="nearest"
    )

# ───────── Training & Eval ─────────
def train_and_eval(proc, lab_par):
    lab = create_target(
        proc.copy(),
        vol_window=lab_par["vol_window"],
        barrier_multiplier=lab_par["barrier_multiplier"],
        cusum_multiplier=lab_par["cusum_multiplier"],
        horizon_seconds=lab_par["horizon_seconds"]
    )
    lab = pd.merge(lab, proc, on="TimeStamp", how="left")

    feats = [
        "sentiment_score","sentiment_score_long",
        "vol_5","vol_20","ret_1","ret_5","hour","minute"
    ]
    X, y = lab[feats], lab["target"]
    Xb, yb = SMOTE(random_state=42).fit_resample(X, y)
    yb_enc = (yb>0).astype(int)

    rf   = RandomForestClassifier(400, n_jobs=1, class_weight="balanced", random_state=42)
    xgbc = xgb.XGBClassifier(tree_method="hist", predictor="cpu_predictor",
                             n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0, random_state=42)
    lgbc = lgb.LGBMClassifier(device_type="cpu", verbose=-1,
                              n_estimators=600, learning_rate=0.05,
                              num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                              class_weight="balanced", random_state=42)
    cat  = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
                              eval_metric="AUC", loss_function="Logloss",
                              verbose=False, task_type="CPU", random_state=42)

    for mdl in (rf, xgbc, lgbc, cat):
        mdl.fit(Xb, yb_enc)

    meta_X = np.column_stack([m.predict_proba(Xb)[:,1] for m in (rf, xgbc, lgbc, cat)])
    meta   = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, class_weight="balanced")
    for s in ("accuracy","roc_auc","average_precision"):
        v = cross_val_score(meta, meta_X, yb_enc, cv=TimeSeriesSplit(KFOLD), scoring=s, n_jobs=1).mean()
        logging.info(f"Meta‑{s}: {v:.4f}")
    meta.fit(meta_X, yb_enc)

    joblib.dump((rf, xgbc, lgbc, cat, meta), MODEL_OUT)
    logging.info("Modelle gespeichert → %s", MODEL_OUT)
    return rf, xgbc, lgbc, cat, meta

# ───────── Walk‑Forward Backtest & Main ─────────
def walk_backtest(models, df, chunk_days=90):
    rf, xgbc, lgbc, cat, meta = models
    df = df.sort_values("TimeStamp")
    end = df["TimeStamp"].min() + timedelta(days=chunk_days)
    perf = []
    while end < df["TimeStamp"].max():
        sub = df[df["TimeStamp"] <= end]
        if len(sub) < 2: break
        feats = sub[["sentiment_score","sentiment_score_long"]]
        basis = np.column_stack([m.predict_proba(feats)[:,1] for m in (rf, xgbc, lgbc, cat)])
        preds = (meta.predict_proba(basis)[:,1] > 0.5).astype(int)*2 - 1
        perf.append((preds * sub["Tick_Last"].pct_change().fillna(0).values).sum())
        end += timedelta(days=chunk_days)
    return np.sum(perf)

def main():
    start = time.time()
    raw   = pd.read_csv(DATA_RAW, encoding="utf-8")
    raw   = standardize_tick_last(ensure_timestamp(raw))
    split = raw["TimeStamp"].quantile(0.9)
    train, test = raw[raw["TimeStamp"] < split], raw[raw["TimeStamp"] >= split]

    logging.info("Sentiment‑Tuning …")
    sent_par = tune_sentiment(train, N_SENT_TRIALS)

    logging.info("Feature‑Engineering …")
    proc_tr = build_feature_df(train, sent_par)
    proc_te = build_feature_df(test,  sent_par)

    logging.info("Label‑Tuning …")
    lab_par = tune_labeling(proc_tr, N_LABEL_TRIALS)

    logging.info("Training & Evaluation …")
    models = train_and_eval(proc_tr, lab_par)

    logging.info("Walk‑Forward Backtest …")
    all_proc = build_feature_df(raw, sent_par)
    bt = walk_backtest(models, all_proc)
    logging.info("Walk‑Forward PnL: %.4f", bt)

    logging.info("Fertig in %s", timedelta(seconds=round(time.time()-start)))

if __name__ == "__main__":
    main()
