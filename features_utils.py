# ────────────────────────────────────────────────────────────
#   ▸ load_ticks            : CSV/Parquet → DataFrame
#   ▸ make_hourly_bars      : Tick-→OHLCV-Resampling (1-H)
#   ▸ triple_barrier_label  : up/down/neutral   (−1/0/+1)
#   ▸ leak_filter           : entfernt Bars, die das zukünftige
#                             Horizon-Fenster überschneiden
# ────────────────────────────────────────────────────────────
# features_utils.py  ──────────────────────────────────────────────
from pathlib import Path          # <─  neu
from datetime import timedelta    # <─  neu
import pandas as pd               # <─  neu
import numpy as np                # <─  neu


def load_ticks(path: Path) -> pd.DataFrame:
    df = (
        pd.read_csv(path, parse_dates=["Time"])    # ← Time → Datum
          .rename(columns={
              "Time":      "TimeStamp",
              "Tick_Bid":  "Bid",
              "Tick_Ask":  "Ask",
              "Tick_Last": "Last",          # ← HIER umbenennen!
          })
          .set_index("TimeStamp")
          .sort_index()
    )
    return df

# ------------------------------------------------------------------
# 1)  make_hourly_bars  – stellt sicher, dass ‚Close‘ existiert
# ------------------------------------------------------------------
def make_hourly_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Tick-Data (Bid/Ask/Last) zu 1-Stunden-OHLC-Bars.
    ▸  liefert Spalten: TimeStamp, Open, High, Low, Close, Volume
    """
    # Wir nehmen den Letztpreis (oder Bid, falls es kein Last gibt)
    price_col = "Last" if "Last" in ticks.columns else "Bid"
    
    ohlc = ticks[price_col].resample("1H").ohlc()
    ohlc.columns = ["Open", "High", "Low", "Close"]        # << Großschreibung!
    
    vol  = ticks[price_col].resample("1H").size().rename("Volume")
    bars = pd.concat([ohlc, vol], axis=1).dropna().reset_index()

    bars.rename(columns={"index": "TimeStamp"}, inplace=True)
    return bars


# ------------------------------------------------------------------
# 2)  triple_barrier_label  – robuster auf Preis-Spalte prüfen
# ------------------------------------------------------------------
N_CLASSES = 3          # long / short / flat (=0)

def triple_barrier_label(df: pd.DataFrame,
                         hor: int,
                         thr_up: float,
                         thr_dn: float) -> np.ndarray:
    """
    Berechnet 3-Klassen-Labels (1=long, 2=short, 0=flat) und
    überspringt Zeilen mit ungültigem Preis.
    """
    close = df["Close"].astype("float64").values
    n      = len(close)
    label  = np.full(n, -1, dtype=np.int8)        # -1 => wird später gedroppt

    # --- optional: Preise prüfen und DataFrame vorher säubern ------------
    bad_mask = np.isnan(close) | (close <= 0)
    if bad_mask.any():
        # betroffene Zeilen direkt aus dem DataFrame werfen
        df.drop(index=df.index[bad_mask], inplace=True)
        close = close[~bad_mask]
        n     = len(close)
        label = np.full(n, -1, dtype=np.int8)

    # --- Barrieren --------------------------------------------------------
    for i in range(n):
        j_end = min(i + hor, n - 1)

        base  = close[i]
        if base == 0 or np.isnan(base):
            continue                                    # überspringen

        # sichere Division ohne Warnung
        with np.errstate(divide="ignore", invalid="ignore"):
            path = (close[i : j_end + 1] - base) / base

        # Trefferzeitpunkte
        hit_up = np.where(path >=  thr_up)[0]
        hit_dn = np.where(path <= -thr_dn)[0]

        t_up = hit_up[0] if hit_up.size else np.inf
        t_dn = hit_dn[0] if hit_dn.size else np.inf

        if t_up < t_dn:
            label[i] = 1          # long
        elif t_dn < t_up:
            label[i] = 2          # short
        else:
            label[i] = 0          # flat / horizon reached

    return label

def leak_filter(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """
    Entfernt alle Zeilen, deren Label-Horizont in die Zukunft ragt,
    die in den aktuellen Trainingsdaten bereits „sichtbar“ wäre.
    """
    # --- Zeitachse sauber bereitstellen ---------------------------------
    if "TimeStamp" in df.columns:
        ts = df["TimeStamp"]
    else:                       # liegt im Index
        ts = df.index.to_series()

    # spätestens zulässiger Startpunkt
    latest_allowed = ts.max() - timedelta(hours=horizon_h)

    mask = ts <= latest_allowed
    return df.loc[mask].copy()

