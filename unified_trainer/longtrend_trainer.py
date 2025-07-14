import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


from .helpers import (
    read_raw_csv,
    make_bars,
    FeatureBuilder,
    split_datasets,
    ts_split,
    triple_barrier_label,
)


def load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw tick CSV/Parquet file."""
    raw = next(read_raw_csv(Path(path)))
    if nrows:
        raw = raw.head(nrows)
    return raw


def main(config_path: str = "config.yaml", nrows: int | None = None) -> None:
    config = yaml.safe_load(open(config_path))
    raw = load_raw(config["raw_tick_path"], nrows=nrows)
    bars = make_bars(raw, freq="1H")
    is_end = bars["timestamp"].iloc[int(len(bars) * 0.6)]
    oos1_end = bars["timestamp"].iloc[int(len(bars) * 0.8)]

    def build_features(trial: optuna.trial.Trial) -> pd.DataFrame:
        fb = FeatureBuilder(bars)
        df_feat = fb.build(
            lags=(1, trial.suggest_int("lag2", 3, 10), 12, trial.suggest_int("lag4", 20, 30)),
            vol_wins=(trial.suggest_int("vs", 10, 30), trial.suggest_int("vl", 50, 120)),
            sma_wins=(trial.suggest_int("sma_f", 10, 30), trial.suggest_int("sma_s", 40, 120)),
            ema_spans=(trial.suggest_int("ema_f", 5, 20), trial.suggest_int("ema_s", 20, 60)),
            rsi_win=trial.suggest_int("rsi_w", 10, 30),
            boll=(trial.suggest_int("boll_w", 15, 40), trial.suggest_float("boll_s", 1.5, 2.5)),
        )
        df_feat["target"] = triple_barrier_label(
            df_feat,
            trial.suggest_int("hor", 1, 3),
            trial.suggest_float("thr_up", 0.002, 0.01),
            trial.suggest_float("thr_dn", 0.002, 0.01),
        )
        df_feat["target"] = (df_feat["target"] == 1).astype(int)
        if df_feat["target"].nunique() < 2:
            df_feat["target"] = (df_feat["close"].shift(-1) > df_feat["close"]).astype(int)
        df_feat.dropna(inplace=True)
        return df_feat

    def objective(trial: optuna.trial.Trial) -> float:
        df_feat = build_features(trial)
        df_is, _, _ = split_datasets(df_feat, is_end, oos1_end)
        y = df_is.pop("target")
        X = df_is.drop(columns=["timestamp"])
        if y.nunique() < 2:
            y.iloc[-1] = 1 - y.iloc[-1]

from .helpers import read_raw_csv, make_bars, FeatureBuilder, split_datasets, ts_split


def load_data(path: str) -> pd.DataFrame:
    raw = next(read_raw_csv(Path(path)))
    bars = make_bars(raw, freq="1H")
    df = FeatureBuilder(bars).build()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df


def main(config_path: str = "config.yaml") -> None:
    config = yaml.safe_load(open(config_path))
    df = load_data(config["raw_tick_path"])
    is_end = df["timestamp"].iloc[int(len(df) * 0.6)]
    oos1_end = df["timestamp"].iloc[int(len(df) * 0.8)]
    df_is, df_oos1, df_oos2 = split_datasets(df, is_end, oos1_end)

    y = df_is.pop("target")
    X = df_is

    def objective(trial: optuna.trial.Trial) -> float:

        hp = {
            "rf_ne": trial.suggest_int("rf_ne", 100, 300),
            "rf_md": trial.suggest_int("rf_md", 3, 10),
            "gb_ne": trial.suggest_int("gb_ne", 100, 300),
            "gb_lr": trial.suggest_float("gb_lr", 0.01, 0.2),

            "meta_c": trial.suggest_float("meta_c", 0.1, 10.0, log=True),

        }
        losses = []
        for tr, va in ts_split(X, config["cv"]["n_splits"]):
            rf = RandomForestClassifier(n_estimators=hp["rf_ne"], max_depth=hp["rf_md"], random_state=config["cv"]["seed"])
            gb = GradientBoostingClassifier(n_estimators=hp["gb_ne"], learning_rate=hp["gb_lr"])

            if len(np.unique(y.iloc[tr])) < 2 or len(np.unique(y.iloc[va])) < 2:
                losses.append(1.0)
                continue

            rf.fit(X.iloc[tr], y.iloc[tr])
            gb.fit(X.iloc[tr], y.iloc[tr])
            stack = np.column_stack([
                rf.predict_proba(X.iloc[va])[:, 1],
                gb.predict_proba(X.iloc[va])[:, 1],
            ])

            meta = LogisticRegression(max_iter=1000, C=hp["meta_c"]).fit(stack, y.iloc[va])

            meta = LogisticRegression(max_iter=1000).fit(stack, y.iloc[va])

            pred = meta.predict_proba(stack)[:, 1]
            losses.append(log_loss(y.iloc[va], pred))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])
    best = study.best_trial.params


    df_feat = build_features(study.best_trial)
    df_is, df_oos1, df_oos2 = split_datasets(df_feat, is_end, oos1_end)
    y = df_is.pop("target")
    X = df_is.drop(columns=["timestamp"])
    if y.nunique() < 2:
        y.iloc[-1] = 1 - y.iloc[-1]


    rf = RandomForestClassifier(n_estimators=best["rf_ne"], max_depth=best["rf_md"], random_state=config["cv"]["seed"])
    gb = GradientBoostingClassifier(n_estimators=best["gb_ne"], learning_rate=best["gb_lr"])
    rf.fit(X, y)
    gb.fit(X, y)
    stack = np.column_stack([rf.predict_proba(X)[:, 1], gb.predict_proba(X)[:, 1]])

    meta = LogisticRegression(max_iter=1000, C=best["meta_c"]).fit(stack, y)

    meta = LogisticRegression(max_iter=1000).fit(stack, y)


    result = {
        "model": {"rf": rf, "gb": gb, "meta": meta},
        "preprocessor": None,
        "config": config,
        "is_dates": (df_is["timestamp"].min(), df_is["timestamp"].max()),
        "oos1_dates": (df_oos1["timestamp"].min(), df_oos1["timestamp"].max()),
        "oos2_dates": (df_oos2["timestamp"].min(), df_oos2["timestamp"].max()),
    }

    with open(config["output_paths"]["longtrend"], "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
