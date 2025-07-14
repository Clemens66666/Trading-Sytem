import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Generator, Iterable, Tuple, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin


def read_raw_csv(path: Path, chunksize: int = 5_000_000) -> Generator[pd.DataFrame, None, None]:

    """Yield normalised CSV chunks (supports ZIP) with lower case columns."""
    for chunk in pd.read_csv(path, chunksize=chunksize, compression="infer"):

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

        for c in (
            "close",
            "bid",
            "ask",
            "last",
            "tick_last",
            "tick_bid",
            "tick_ask",
        ):

        for c in ("close", "bid", "ask", "last"):

            if c in df.columns:
                price_col = c
                break
    ohlc = df[price_col].resample(freq).ohlc()
    vol = df.get("volume", df.get("tick_volume", pd.Series(1, index=df.index)))
    bars = ohlc.join(vol.resample(freq).sum().rename("volume"))
    return bars.dropna().reset_index().rename(columns={ts_col: "timestamp"})


class FeatureBuilder:

    """Simple technical feature calculator for bar data.

    This builder exposes many optional feature groups that can be toggled or
    parametrised. The default settings result in more than 25 numeric features
    which are sufficient for basic ML models.
    """

    """Simple technical feature calculator for bar data."""


    def __init__(self, bars: pd.DataFrame):
        self.df = bars.copy()

    def add_basic(self) -> "FeatureBuilder":
        self.df["vwap"] = self.df[["open", "high", "low", "close"]].mean(1)
        self.df["bar_range"] = self.df["high"] - self.df["low"]
        self.df["bar_return"] = self.df["close"].pct_change().fillna(0)
        return self


    def add_lags(self, lags: Iterable[int] = (1, 5, 12, 24)) -> "FeatureBuilder":

    def add_lags(self, lags: Iterable[int] = (5, 12)) -> "FeatureBuilder":

        for l in lags:
            self.df[f"close_lag{l}"] = self.df["close"].shift(l)
        return self


    def add_volume(self, wins: Tuple[int, int] = (12, 72)) -> "FeatureBuilder":
        if "volume" not in self.df.columns:
            self.df["volume"] = 0.0
        short, long = wins
        self.df["vol_short"] = self.df["volume"].rolling(short).sum().fillna(0)
        self.df["vol_long"] = self.df["volume"].rolling(long).sum().fillna(0)
        return self

    def add_returns(self, wins: Tuple[int, int] = (1, 5)) -> "FeatureBuilder":
        s1, s5 = wins
        self.df["ret_1"] = self.df["close"].pct_change(s1).fillna(0)
        self.df["ret_5"] = self.df["close"].pct_change(s5).fillna(0)
        return self

    def add_sma(self, wins: Tuple[int, int] = (12, 72)) -> "FeatureBuilder":
        fast, slow = wins
        self.df["sma_fast"] = self.df["close"].rolling(fast).mean().bfill()
        self.df["sma_slow"] = self.df["close"].rolling(slow).mean().bfill()
        return self

    def add_ema(self, spans: Tuple[int, int] = (12, 26)) -> "FeatureBuilder":
        fast, slow = spans
        self.df["ema_fast"] = self.df["close"].ewm(span=fast, adjust=False).mean()
        self.df["ema_slow"] = self.df["close"].ewm(span=slow, adjust=False).mean()
        return self

    def add_rsi(self, win: int = 14) -> "FeatureBuilder":
        d = self.df["close"].diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        rs = up.ewm(alpha=1 / win, adjust=False).mean() / (dn.ewm(alpha=1 / win, adjust=False).mean() + 1e-8)
        self.df["rsi"] = 100 - 100 / (1 + rs)
        return self

    def add_macd(self, fast: int = 12, slow: int = 26, sig: int = 9) -> "FeatureBuilder":
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        self.df["macd"] = macd
        self.df["macd_signal"] = macd.ewm(span=sig, adjust=False).mean()
        return self

    def add_bollinger(self, win: int = 20, n_std: float = 2.0) -> "FeatureBuilder":
        ma = self.df["close"].rolling(win).mean()
        std = self.df["close"].rolling(win).std()
        self.df["boll_up"] = ma + n_std * std
        self.df["boll_dn"] = ma - n_std * std
        return self

    def add_ofi(self) -> "FeatureBuilder":
        if "volume" not in self.df.columns:
            self.df["volume"] = 0.0
        self.df["ofi_min"] = self.df["close"].diff().fillna(0) * self.df["volume"]
        return self

    def build(
        self,
        lags: Iterable[int] = (1, 5, 12, 24),
        vol_wins: Tuple[int, int] = (12, 72),
        ret_wins: Tuple[int, int] = (1, 5),
        sma_wins: Tuple[int, int] = (12, 72),
        ema_spans: Tuple[int, int] = (12, 26),
        rsi_win: int = 14,
        macd: Tuple[int, int, int] = (12, 26, 9),
        boll: Tuple[int, float] = (20, 2.0),
    ) -> pd.DataFrame:
        """Return dataframe with engineered features."""
        f_macd, s_macd, sig = macd
        b_win, b_std = boll
        return (
            self.add_basic()
            .add_lags(lags)
            .add_volume(vol_wins)
            .add_returns(ret_wins)
            .add_sma(sma_wins)
            .add_ema(ema_spans)
            .add_rsi(rsi_win)
            .add_macd(f_macd, s_macd, sig)
            .add_bollinger(b_win, b_std)
            .add_ofi()
            .df.fillna(0)
        )

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



def high_low_trend_label(
    df: pd.DataFrame,
    window: int,
    trend: pd.Series | None = None,
    up_thresh: float = 0.55,
    down_thresh: float = 0.45,
) -> np.ndarray:
    """Label highs/lows within a rolling window optionally conditioned on trend.

    Parameters
    ----------
    df : DataFrame
        Must contain ``timestamp``, ``high`` and ``low`` columns.
    window : int
        Number of bars to look back (e.g. ``3`` for 30 minutes on 10‑minute bars).
    trend : Series, optional
        If provided, long entries (1) are only labelled when ``trend >= up_thresh``
        and short entries (0) when ``trend <= down_thresh``.

    Returns
    -------
    np.ndarray
        Array with 1 for lows, 0 for highs and -1 otherwise.
    """
    highs = df["high"].rolling(window, min_periods=window).max()
    lows = df["low"].rolling(window, min_periods=window).min()
    is_high = df["high"] >= highs
    is_low = df["low"] <= lows
    labels = np.full(len(df), -1, dtype=np.int8)

    if trend is not None:
        tr = trend.reindex(df["timestamp"], method="ffill").to_numpy()
        labels[(is_low) & (tr >= up_thresh)] = 1
        labels[(is_high) & (tr <= down_thresh)] = 0
    else:
        labels[is_low] = 1
        labels[is_high] = 0

    labels[is_high & is_low] = -1
    return labels


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



def stack_predict(model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """Return stacked probability for class 1 from rf+gb meta ensemble."""
    rf = model_dict.get("rf")
    gb = model_dict.get("gb")
    meta = model_dict.get("meta")
    arr = X.to_numpy()
    stack = np.column_stack([
        rf.predict_proba(arr)[:, 1],
        gb.predict_proba(arr)[:, 1],
    ])
    return meta.predict_proba(stack)[:, 1]



def split_datasets(df: pd.DataFrame, is_end_date: pd.Timestamp, oos1_end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into IS, OOS1 and OOS2 segments by date."""
    df = ensure_timestamp(df).sort_values("timestamp")
    df_is = df[df["timestamp"] <= is_end_date]
    df_oos1 = df[(df["timestamp"] > is_end_date) & (df["timestamp"] <= oos1_end_date)]
    df_oos2 = df[df["timestamp"] > oos1_end_date]
    return df_is, df_oos1, df_oos2
