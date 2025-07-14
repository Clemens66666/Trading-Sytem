import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .helpers import read_raw_csv, make_bars, FeatureBuilder, split_datasets, ts_split


def load_data(path: str) -> pd.DataFrame:
    raw = next(read_raw_csv(Path(path)))
    bars = make_bars(raw, freq="5T")
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
        }
        losses = []
        for tr, va in ts_split(X, config["cv"]["n_splits"]):
            rf = RandomForestClassifier(n_estimators=hp["rf_ne"], max_depth=hp["rf_md"], random_state=config["cv"]["seed"])
            gb = GradientBoostingClassifier(n_estimators=hp["gb_ne"], learning_rate=hp["gb_lr"])
            rf.fit(X.iloc[tr], y.iloc[tr])
            gb.fit(X.iloc[tr], y.iloc[tr])
            stack = np.column_stack([
                rf.predict_proba(X.iloc[va])[:, 1],
                gb.predict_proba(X.iloc[va])[:, 1],
            ])
            meta = LogisticRegression(max_iter=1000).fit(stack, y.iloc[va])
            pred = meta.predict_proba(stack)[:, 1]
            losses.append(log_loss(y.iloc[va], pred))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])
    best = study.best_trial.params

    rf = RandomForestClassifier(n_estimators=best["rf_ne"], max_depth=best["rf_md"], random_state=config["cv"]["seed"])
    gb = GradientBoostingClassifier(n_estimators=best["gb_ne"], learning_rate=best["gb_lr"])
    rf.fit(X, y)
    gb.fit(X, y)
    stack = np.column_stack([rf.predict_proba(X)[:, 1], gb.predict_proba(X)[:, 1]])
    meta = LogisticRegression(max_iter=1000).fit(stack, y)

    result = {
        "model": {"rf": rf, "gb": gb, "meta": meta},
        "preprocessor": None,
        "config": config,
        "is_dates": (df_is["timestamp"].min(), df_is["timestamp"].max()),
        "oos1_dates": (df_oos1["timestamp"].min(), df_oos1["timestamp"].max()),
        "oos2_dates": (df_oos2["timestamp"].min(), df_oos2["timestamp"].max()),
    }

    with open(config["output_paths"]["exit"], "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
