import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Generator, Iterable, Tuple, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin


def read_raw_csv(path: Path, chunksize: int = 5_000_000) -> Generator[pd.DataFrame, None, None]:
    """Yield normalized CSV chunks with lower-case column names."""
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
        yield chunk


def make_bars(df: pd.DataFrame, freq: str = "5T") -> pd.DataFrame:
    """Resample a tick DataFrame to OHLCV bars."""
    df = df.copy()
    ts_col = df.filter(like="time").columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df.set_index(ts_col, inplace=True)
    price_col = "price"
    if price_col not in df.columns:
        for c in ("close", "bid", "ask", "last"):
            if c in df.columns:
                price_col = c
                break
    ohlc = df[price_col].resample(freq).ohlc()
    vol = df.get("volume", df.get("tick_volume", pd.Series(1, index=df.index)))
    bars = ohlc.join(vol.resample(freq).sum().rename("volume"))
    return bars.dropna().reset_index().rename(columns={ts_col: "timestamp"})


class FeatureBuilder:
    """Simple technical feature calculator for bar data."""

    def __init__(self, bars: pd.DataFrame):
        self.df = bars.copy()

    def add_basic(self) -> "FeatureBuilder":
        self.df["vwap"] = self.df[["open", "high", "low", "close"]].mean(1)
        self.df["bar_range"] = self.df["high"] - self.df["low"]
        self.df["bar_return"] = self.df["close"].pct_change().fillna(0)
        return self

    def add_lags(self, lags: Iterable[int] = (5, 12)) -> "FeatureBuilder":
        for l in lags:
            self.df[f"close_lag{l}"] = self.df["close"].shift(l)
        return self

    def build(self) -> pd.DataFrame:
        return self.add_basic().add_lags().df.fillna(0)


class MarketSentimentTransformer(BaseEstimator, TransformerMixin):
    """Rolling sentiment score based on price and volume."""

    def __init__(self, window: int = 100, w1: float = 0.3, w2: float = 0.5, w3: float = 0.2):
        self.window, self.w1, self.w2, self.w3 = window, w1, w2, w3

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["ret"] = df["close"].pct_change().fillna(0)
        df["mom"] = df["close"] - df["close"].shift(self.window)
        vol = df["close"].rolling(self.window, min_periods=1).std().fillna(0)
        volC = df.get("volume", 1).rolling(self.window, min_periods=1).mean()
        volC = df.get("volume", 1) / (volC + 1e-6) - 1
        df["sentiment_score"] = (self.w1*df["ret"] + self.w2*df["mom"] + self.w3*volC) / (vol + 1e-6)
        return df


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Adds volatility and return based features."""

    def __init__(self, win_short: int = 5, win_long: int = 20):
        self.win_short, self.win_long = win_short, win_long

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["vol_short"] = df["close"].rolling(self.win_short).std().fillna(0)
        df["vol_long"] = df["close"].rolling(self.win_long).std().fillna(0)
        df["ret_1"] = df["close"].pct_change().fillna(0)
        df["ret_5"] = df["close"].pct_change(5).fillna(0)
        return df.fillna(0)


def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'timestamp' column exists and is datetime."""
    if "timestamp" not in df.columns:
        if "TimeStamp" in df.columns:
            df = df.rename(columns={"TimeStamp": "timestamp"})
        elif df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
        else:
            raise KeyError("No timestamp column found")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def resample_to_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample ticks to bars using pandas rule string."""
    df = ensure_timestamp(df).set_index("timestamp").sort_index()
    ohlc = df["close"].resample(rule).ohlc()
    vol = df.get("volume", pd.Series(1, index=df.index)).resample(rule).sum()
    bars = ohlc.join(vol.rename("volume"))
    return bars.dropna().reset_index().rename(columns={"index": "timestamp"})


def compute_ofi_min(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute order flow imbalance and resample to minutes."""
    df = ensure_timestamp(raw)
    df = df.sort_values("timestamp")
    df["tick_sign"] = np.sign(df["close"].diff()).fillna(0)
    df["ofi_raw"] = df["tick_sign"] * df.get("volume", 1)
    ofi = (
        df.set_index("timestamp")["ofi_raw"].rolling(window, min_periods=1).sum()
        / (df.get("volume", 1).rolling(window, min_periods=1).sum() + 1e-6)
    )
    return ofi.resample("1T").last().ffill().rename("ofi_min").reset_index()


def triple_barrier_label(df: pd.DataFrame, hor: int, thr_up: float, thr_dn: float) -> np.ndarray:
    """Compute three-class barrier label."""
    close = df["close"].to_numpy(float)
    n = len(close)
    label = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        j_end = min(i + hor, n - 1)
        base = close[i]
        path = (close[i:j_end+1] - base) / base
        hit_up = np.where(path >= thr_up)[0]
        hit_dn = np.where(path <= -thr_dn)[0]
        t_up = hit_up[0] if hit_up.size else np.inf
        t_dn = hit_dn[0] if hit_dn.size else np.inf
        if t_up < t_dn:
            label[i] = 1
        elif t_dn < t_up:
            label[i] = 2
        else:
            label[i] = 0
    return label


def leak_filter(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Remove rows that would leak future horizon."""
    df = ensure_timestamp(df)
    latest_allowed = df["timestamp"].max() - timedelta(hours=horizon_h)
    return df[df["timestamp"] <= latest_allowed].copy()


class CustomPurgedKFold:
    """Purged K-Fold with embargo period."""

    def __init__(self, n_splits: int = 5, samples_info_sets: pd.Series = None, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo = int(n * self.pct_embargo)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        sis = self.samples_info_sets.reset_index(drop=True)
        for tr, te in tscv.split(np.arange(n)):
            test_times = sis.iloc[te]
            start, end = test_times.min(), test_times.max()
            purged = [i for i in tr if sis.iat[i] < start or sis.iat[i] > end]
            emb_start = te.max() + 1
            emb_end = min(n, emb_start + embargo)
            final_train = [i for i in purged if i < emb_start or i >= emb_end]
            yield final_train, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def ts_split(X, n_splits: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Generator yielding train/val indices for time series CV."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tss.split(X):
        yield tr, va


def filter_valid_estimators(models: dict) -> List[Tuple[str, BaseEstimator]]:
    """Return estimators that implement predict_proba."""
    valid = []
    for name, est in models.items():
        if est is None or not hasattr(est, "predict_proba"):
            continue
        valid.append((name, est))
    return valid


def split_datasets(df: pd.DataFrame, is_end_date: pd.Timestamp, oos1_end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into IS, OOS1 and OOS2 segments by date."""
    df = ensure_timestamp(df).sort_values("timestamp")
    df_is = df[df["timestamp"] <= is_end_date]
    df_oos1 = df[(df["timestamp"] > is_end_date) & (df["timestamp"] <= oos1_end_date)]
    df_oos2 = df[df["timestamp"] > oos1_end_date]
    return df_is, df_oos1, df_oos2
