import logging
import pandas as pd
import numpy as np
from datetime import timedelta

from trainer_globals import BAR_RULE_MIN, BAR_RULE_HOUR, EPS

# ───────── Helpers ─────────

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass es eine 'TimeStamp' Spalte gibt und
    konvertiert sie in datetime. Akzeptiert außerdem DataFrames
    mit DatetimeIndex (wird dann in eine Spalte umgewandelt).
    """
    # Fall: DatetimeIndex → Spalte
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "TimeStamp"})
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
        return df

    # Fall: schon Spalte vorhanden
    if "TimeStamp" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    elif "Time" in df.columns:
        df = df.rename(columns={"Time": "TimeStamp"})
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    else:
        raise ValueError("Keine Zeitspalte gefunden (erwartet 'TimeStamp' oder 'Time').")
    return df

def standardize_tick_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sucht in den Spalten nach dem letzten Kurs und setzt 'Tick_Last'.
    """
    if "Tick_Last" in df.columns:
        return df
    if "close" in df.columns:
        df["Tick_Last"] = df["close"]
        return df
    for c in df.columns:
        if "price" in c.lower():
            df["Tick_Last"] = df[c]
            return df
    raise ValueError("Keine Preis-Spalte gefunden!")

def resample_to_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resampled Tick-Daten zu OHLCV-Bars nach dem pandas-Rule-String.
    Fügt 'vwap', 'bar_range', 'bar_return' hinzu.
    """
    df = ensure_timestamp(df).copy()
    df = standardize_tick_last(df)
    df = df.set_index("TimeStamp").sort_index()
    # OHLC
    ohlc = df["Tick_Last"].resample(rule).ohlc()
    # Volumen
    vol = df.get("Tick_Volume", pd.Series(1.0, index=df.index)).resample(rule).sum()
    # VWAP
    vwap_num = (df["Tick_Last"] * df.get("Tick_Volume", 1.0)).resample(rule).sum()
    bars = ohlc.join(vol.rename("Tick_Volume"))
    bars["vwap"] = vwap_num / (bars["Tick_Volume"] + EPS)
    bars = bars.dropna()
    # Zusatz-Features
    bars["bar_range"]  = bars["high"] - bars["low"]
    bars["bar_return"] = (bars["close"] - bars["open"]) / (bars["open"] + EPS)
    bars["Tick_Last"]  = bars["close"]
    return bars.reset_index()

def compute_ofi_min(raw: pd.DataFrame, ofi_window: int) -> pd.DataFrame:
    """
    Berechnet Order-Flow-Imbalance (OFI) auf Tick-Level
    und resampled das Ergebnis auf BAR_RULE_MIN.
    """
    df = ensure_timestamp(raw.copy())
    df = standardize_tick_last(df)
    df["Tick_Volume"] = df.get("Tick_Volume", 1.0)
    df = df.sort_values("TimeStamp")
    df["tick_sign"] = np.sign(df["Tick_Last"].diff()).fillna(0)
    df["ofi_raw"]   = df["tick_sign"] * df["Tick_Volume"]
    df["ofi_tick"]  = (
        df["ofi_raw"].rolling(window=ofi_window, min_periods=1).sum()
        / (df["Tick_Volume"].rolling(window=ofi_window, min_periods=1).sum() + EPS)
    )
    ofi_min = (
        df.set_index("TimeStamp")["ofi_tick"]
          .resample(BAR_RULE_MIN).last()
          .ffill()
          .rename("ofi_min")
          .reset_index()
    )
    return ofi_min

# ───────── Transformer ─────────

from sklearn.base import BaseEstimator, TransformerMixin

class MarketSentimentTransformer(BaseEstimator, TransformerMixin):
    """
    Berechnet den Sentiment-Score über ein Fenster
    mit gewichteter Kombination von Return/Momentum/VolC.
    """
    def __init__(self, window=100, w1=0.3, w2=0.5, w3=0.2):
        self.window, self.w1, self.w2, self.w3 = window, w1, w2, w3

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kw):
        df = ensure_timestamp(X.copy())
        df = standardize_tick_last(df)
        df["Tick_Volume"] = df.get("Tick_Volume", 1.0)
        ret  = df["Tick_Last"].pct_change().fillna(0)
        mom  = (df["Tick_Last"] - df["Tick_Last"].shift(self.window)).fillna(0)
        vol  = df["Tick_Last"].rolling(self.window, min_periods=1).std().fillna(EPS)
        volC = df["Tick_Volume"] / (df["Tick_Volume"].rolling(self.window, min_periods=1).mean() + EPS) - 1
        raw  = (self.w1*ret + self.w2*mom + self.w3*volC) / (vol + EPS)
        df["sentiment_score"] = raw
        return df

class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Fügt technische Modern-Features hinzu: Volatility, Returns,
    RSI, MACD, OBV, Bar-Range/Return, VWAP-Offset, Volume-Spike.
    """
    def __init__(self, vs=5, vl=20, r1=1, r5=5, rsi_w=14, mf=12, ms=26, msig=9):
        self.vol_short, self.vol_long = vs, vl
        self.ret1, self.ret5 = r1, r5
        self.rsi_w, self.mf, self.ms, self.msig = rsi_w, mf, ms, msig
        self.eps = EPS

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kw):
        df = X.copy()
        # Rolling-Volatility
        df["vol_5"]  = df["Tick_Last"].rolling(self.vol_short).std().fillna(0)
        df["vol_20"] = df["Tick_Last"].rolling(self.vol_long).std().fillna(0)
        # Returns
        df["ret_1"]  = df["Tick_Last"].pct_change(self.ret1).fillna(0)
        df["ret_5"]  = df["Tick_Last"].pct_change(self.ret5).fillna(0)
        # Uhrzeit
        df["hour"]   = df["TimeStamp"].dt.hour
        df["minute"] = df["TimeStamp"].dt.minute
        # RSI
        delta    = df["Tick_Last"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_w).mean()
        avg_loss = loss.rolling(self.rsi_w).mean().replace(0, np.nan)
        rs       = avg_gain / avg_loss
        df["rsi"] = (100 - 100/(1+rs)).fillna(0)
        # MACD
        exp1 = df["Tick_Last"].ewm(span=self.mf, adjust=False).mean()
        exp2 = df["Tick_Last"].ewm(span=self.ms, adjust=False).mean()
        macd = exp1 - exp2
        df["macd"]        = macd.fillna(0)
        df["macd_signal"] = macd.ewm(span=self.msig, adjust=False).mean().fillna(0)
        # OBV
        obv = (np.sign(df["Tick_Last"].diff()) * df.get("Tick_Volume",1)).fillna(0)
        df["obv"] = obv.cumsum()
        # Bar-Merges (falls schon da)
        if "bar_range" in df:    df["bar_range"]   = df["bar_range"]
        if "bar_return" in df:   df["bar_return"]  = df["bar_return"]
        if "vwap" in df:         df["vwap_offset"] = df["Tick_Last"] - df["vwap"]
        # Volume-Spike
        df["vol_spike"] = df["Tick_Volume"] / (
            df["Tick_Volume"].rolling(self.vol_long).mean() + self.eps
        ) - 1
        return df.fillna(0)

# ───────── Label-Creator ─────────

def create_target(df: pd.DataFrame,
                  vol_window: int,
                  barrier_multiplier: float,
                  cusum_multiplier: float,
                  horizon_seconds: int,
                  min_gap_seconds: int = 60) -> pd.DataFrame:
    """
    Triple-Barrier + CUSUM-Labeling.
    Liefert DataFrame mit ['TimeStamp','target','price0', ... alle Feature-Spalten].
    """
    # Falls wir nur einen DatetimeIndex haben, als Spalte übernehmen
    if "TimeStamp" not in df.columns and "Time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "TimeStamp"})

    ts = ensure_timestamp(df.copy())
    ts = ts.sort_values("TimeStamp").set_index("TimeStamp")
    ts = standardize_tick_last(ts).loc[~ts.index.duplicated(keep="first")]
    ts["returns"]    = ts["Tick_Last"].pct_change()
    ts["volatility"] = ts["returns"].ewm(span=vol_window, adjust=False).std()

    threshold = cusum_multiplier * ts["volatility"].rolling(vol_window, min_periods=1).mean()

    events, s_pos, s_neg = [], 0.0, 0.0
    for i in range(1, len(ts)):
        diff = ts["Tick_Last"].iat[i] - ts["Tick_Last"].iat[i-1]
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        if s_pos > threshold.iat[i] or s_neg < -threshold.iat[i]:
            events.append(ts.index[i])
            s_pos, s_neg = 0.0, 0.0

    records, last = [], None
    for t in events:
        if last and (t - last).total_seconds() < min_gap_seconds:
            continue
        last = t
        p0 = float(ts.at[t, "Tick_Last"])
        v0 = float(ts.at[t, "volatility"])
        barrier = barrier_multiplier * v0 * p0
        window = ts.loc[t : t + timedelta(seconds=horizon_seconds), "Tick_Last"].to_numpy()
        up_hit = (window >= p0 + barrier).any()
        dn_hit = (window <= p0 - barrier).any()
        lbl = 1 if up_hit else -1 if dn_hit else 0
        if lbl != 0:
            records.append({"TimeStamp": t, "target": lbl, "price0": p0})

    df_ev = pd.DataFrame(records)
    if df_ev.empty:
        logging.warning("[create_target] Keine Events gefunden!")
        return pd.DataFrame(columns=["TimeStamp","target","price0"])

    return df_ev.merge(
        ts.reset_index().drop(columns=["Tick_Last"]),
        on="TimeStamp", how="left"
    )

# ───────── Feature-Building ─────────

def build_feature_df(raw: pd.DataFrame,
                     sent_par: dict,
                     feat_par: dict) -> pd.DataFrame:
    """
    Baut aus Roh-Tick-Daten das vollständige Feature-DF:
    1) 1-Min Bars
    2) OFI-Merge
    3) Lang-Sentiment
    4) Technische Features
    """
    df = ensure_timestamp(raw.copy())
    df = standardize_tick_last(df)
    # 1) Minuten-Bars
    df = resample_to_bars(df, BAR_RULE_MIN)
    # 2) OFI
    ofi_w = feat_par.get("ofi_window", 60)
    ofi   = compute_ofi_min(raw, ofi_w)
    df = pd.merge_asof(
        df.sort_values("TimeStamp"),
        ofi[["TimeStamp","ofi_min"]].sort_values("TimeStamp"),
        on="TimeStamp", direction="backward"
    ).fillna(method="ffill")
    # 3) Lang-Sentiment
    mst = MarketSentimentTransformer(
        window=sent_par["long_window"],
        w1=sent_par["r1"], w2=sent_par["r2"], w3=sent_par["r3"]
    )
    hours = resample_to_bars(raw, BAR_RULE_HOUR)
    sentd = mst.transform(hours)[["TimeStamp","sentiment_score"]].rename(
        columns={"sentiment_score":"sentiment_score_long"}
    )
    df = pd.merge_asof(
        df.sort_values("TimeStamp"),
        sentd.sort_values("TimeStamp"),
        on="TimeStamp", direction="backward"
    )
    # 4) Technische Features
    fe = FeatureAugmenter(
        vs   = feat_par.get("vol_short", 5),
        vl   = feat_par.get("vol_long", 20),
        r1   = feat_par.get("ret1", 1),
        r5   = feat_par.get("ret5", 5),
        rsi_w= feat_par.get("rsi_w", 14),
        mf   = feat_par.get("mf", 12),
        ms   = feat_par.get("ms", 26),
        msig = feat_par.get("msig",9)
    )
    df = fe.transform(df)
    return df.fillna(0)
