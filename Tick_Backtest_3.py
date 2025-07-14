#!/usr/bin/env python3
# ───────────────── backtest_threshold_optuna_fixed.py ─────────────
"""
Full Tick-Backtest + Optuna-Threshold-Suche  (Mapping-Fix + PF-Sharpe)
"""

# ═══════════════════════════════════════════════════════════════════
# 0) KONFIGURATION
# ═══════════════════════════════════════════════════════════════════
RAW_TICKS       = "E:/RawTickData3.txt"        # Tick-CSV / Parquet / gz
LONG_MODEL_PKL  = "E:/longtrend_ensemble_dl.pkl"      # Trend-Ensemble
ENTRY_MODEL_PKL = "E:/entry_stack_fullopt1min.pkl"        # Entry-Stacking
CACHE_PARQUET   = "E:/cache_backtest_1m.parquet"

TICK_STEP       = 1
CHUNK           = 10_000_000
OFI_WINDOW      = 46
BAR_10M, BAR_1H = "1T", "1H"

MIN_LONG , MIN_SHORT =  2000,  2000      # Mindest-Trades pro Seite
N_TRIALS  , N_JOBS   = 2500,  max(4, __import__("multiprocessing").cpu_count()//2)

LOT_SIZE   = 0.01     # Lots per Trade
WAIT_SECS  = 60.0     # Cool-Down Zeit nach Exit (Sekunden)

# ─── Markt- & Kostenparameter ─────────────────────────────────────
PIP              = 0.00010          # 1 Pip in Preis­einheiten
SPREAD_PIP       = 0.12             # Ø-Spread   (0.12 Pip RAW-Konto)
SPREAD_PRICE     = SPREAD_PIP * PIP # 0.000012
COMMISSION_PIP   = 0.60             # Kommission pro *Round-Trip* in Pips
COMMISSION_PER_LOT = COMMISSION_PIP * PIP   # 0.00006 = 0.6 Pip

CONTRACT_SIZE      = 100_000        # 1 Lot EURUSD = 100 000 €
LOT_SIZE           = 0.10           # 0.01 Lot = 1 000 €
SPREAD_PRICE       = 0.00001        # 0.1 Pip
COMMISSION_PER_LOT = 4.0            # Round-Trip (Open+Close)


# ═══════════════════════════════════════════════════════════════════
# 1) IMPORTS
# ═══════════════════════════════════════════════════════════════════
import logging, warnings, numpy as np, pandas as pd
from pathlib import Path
import joblib, numba, optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from tqdm.auto import tqdm
from optuna.storages import RDBStorage

# ══════════════ 1) IMPORTS ═════════════════════════════════════════
import os, warnings, logging, itertools
from pathlib import Path

import numpy as np
import pandas as pd

import joblib, optuna, numba
from optuna.pruners import SuccessiveHalvingPruner
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

# ─── Helper-Funktionen aus dem Training (kopiert) ────────────────
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

def compute_ofi_min(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    df = ensure_timestamp(raw).pipe(standardize_tick_last)
    df["Tick_Volume"] = df.get("Tick_Volume", 1.0)
    df = df.sort_values("TimeStamp")
    df["tick_sign"] = np.sign(df["Tick_Last"].diff()).fillna(0)
    df["ofi_raw"]   = df["tick_sign"] * df["Tick_Volume"]
    df["ofi_tick"]  = df["ofi_raw"].rolling(window, 1).sum() / \
                      (df["Tick_Volume"].rolling(window, 1).sum() + EPS)
    return (df.set_index("TimeStamp")["ofi_tick"]
              .resample(BAR_10M).last().ffill()
              .rename("ofi_min").reset_index())

class MarketSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window=57, w1=.8, w2=.1, w3=.1):
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
    def __init__(self, vs=5, vl=101, r1=3, r5=7,
                 rsi_w=30, mf=6, ms=27, msig=37):
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


def compute_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return 1-hour OHLCV bars just like ``make_hourly_bars`` used in training."""
    df = ensure_timestamp(df)

    price_col = None
    for col in ["Bid", "Tick_Last", "Last", "price", "close", "Close"]:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        raise ValueError("Bid column required for hourly features")

    df = df.rename(columns={price_col: "Bid"})
    df = df.set_index("TimeStamp").sort_index()

    ohlc = df["Bid"].resample(BAR_1H).ohlc()
    ohlc.columns = ["Open", "High", "Low", "Close"]
    vol = df["Bid"].resample(BAR_1H).size().rename("Volume")

    hours = pd.concat([ohlc, vol], axis=1).dropna().reset_index()
    return hours


EPS = 1e-6
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 2) 10-MIN-BARS laden/erzeugen
# ═══════════════════════════════════════════════════════════════════
def build_or_load_10m() -> pd.DataFrame:
    if Path(CACHE_PARQUET).exists():
        log.info("1-Min-Cache gefunden – lade Parquet …")
        return pd.read_parquet(CACHE_PARQUET)

    log.info("Erzeuge 1-Min-Bars (einmalig) …")
    mst    = MarketSentimentTransformer(57, .8, .1, .1)
    fe_aug = FeatureAugmenter(vs=5, vl=101, r1=3, r5=7,
                              rsi_w=30, mf=6, ms=27, msig=37)

    pieces, cols = [], None
    for chunk in tqdm(pd.read_csv(RAW_TICKS, chunksize=CHUNK,
                                  usecols=cols, low_memory=False)):
        chunk = chunk.iloc[::TICK_STEP]
        chunk = ensure_timestamp(chunk).pipe(standardize_tick_last)

        mins  = resample_to_bars(chunk, BAR_10M)
        ofi   = compute_ofi_min(chunk, OFI_WINDOW)
        mins  = pd.merge_asof(mins.sort_values("TimeStamp"), ofi,
                              on="TimeStamp", direction="backward")

        hours = resample_to_bars(chunk, BAR_1H)
        sent  = mst.transform(hours)[["TimeStamp", "sentiment_score"]]
        mins  = pd.merge_asof(mins.sort_values("TimeStamp"), sent,
                              on="TimeStamp", direction="backward")

        mins  = fe_aug.transform(mins).fillna(0.0)
        mins["ATR"] = mins["close"].rolling(14).std().bfill()
        pieces.append(mins)

    candles = (pd.concat(pieces, ignore_index=True)
                 .dropna(subset=["close"])
                 .reset_index(drop=True))
    candles.to_parquet(CACHE_PARQUET, compression="zstd")
    return candles

# ═══════════════════════════════════════════════════════════════════
def pips(n):                # wandelt Pip-Werte in Preis­einheiten
    return n * 0.00010      # EURUSD-Pip = 1e-4

SPREAD_PRICE   = pips(0.1)      # 0.1 Pip  ➜ 0.00001
SL_LONG_FACTOR = pips(10)       # Beispiel: 10 Pips Stopp = 0.0010

# ═══════════════════════════════════════════════════════════════════
# 3) MODELLE & PROBAS  (inkl. Mapping-Fix + Spalten-Check)
# ═══════════════════════════════════════════════════════════════════
def prepare_models(candles: pd.DataFrame):
    log.info("Lade Trend- & Entry-Modelle …")

    # ───────── Long-Trend-Ensemble ─────────────────────────────────
    trend_dict = joblib.load(LONG_MODEL_PKL)
    BASES      = trend_dict["base_models"]
    META_LT    = trend_dict["meta"]
    FEAT_COLS  = [c for c in trend_dict["feat_cols"]
                  if c.lower() != "timestamp"]

    # ───────── Entry-Stacker ───────────────────────────────────────
    entry_obj = joblib.load(ENTRY_MODEL_PKL)
    ENTRY_CLF = entry_obj["model"] if isinstance(entry_obj, dict) else entry_obj

    # ─── 1) Trend-Probability pro Stunde  ──────────────────────────
    hours = compute_hourly_features(
        candles[["TimeStamp", "close"]].rename(columns={"close": "Bid"}))

    X_h     = hours[FEAT_COLS].fillna(0.0).values
    base_p  = np.stack([m.predict_proba(X_h)[:, 1] for m in BASES], axis=1).mean(1)
    trend_p = META_LT.predict_proba(np.column_stack([1.0 - base_p, base_p]))[:, 1]

    trend_ser = pd.Series(trend_p, index=hours["TimeStamp"])
    candles["trend_prob_long"] = trend_ser.reindex(
        candles["TimeStamp"], method="ffill").values

    # jetzt schon das Trend-Array, damit es _vor_ return existiert
    TREND = candles["trend_prob_long"].to_numpy("f4")

    # ─── 2) Entry-Feature-Matrix  (17 exakt definierte Spalten) ───
    ENTRY_FEAT_ORDER = [
        "open", "high", "low", "close", "vwap",
        "bar_range", "bar_return", "ofi_min", "sentiment_score",
        "vol_short", "vol_long", "ret_1", "ret_5", "rsi",
        "macd", "macd_signal", "trend_prob_long"     # ← 17
    ]

    Xp_df = candles.copy()

    missing = [c for c in ENTRY_FEAT_ORDER if c not in Xp_df.columns]
    extras  = [c for c in Xp_df.columns if c not in ENTRY_FEAT_ORDER]

    print("\n=== Spalten-Check ====================")
    print("Fehlende   :", missing)
    print("Unerwartete:", extras[:10], "…" if len(extras) > 10 else "")
    print("======================================\n")

    assert not missing, (
        "Es fehlen Features – siehe Ausgabe oben!"
    )

    Xp_df = Xp_df[ENTRY_FEAT_ORDER]                # exakt 17 Spalten
    Xp    = Xp_df.fillna(0.0).values.astype("f4")

    # ─── 3) Entry-Probas mit MAPPING-PATCH -------------------------
    proba   = ENTRY_CLF.predict_proba(Xp)
    P_LONG  = proba[:, 1].astype("f4")   # Klasse 1 = LONG
    P_SHORT = proba[:, 0].astype("f4")   # Klasse 0 = SHORT

    # ─── 4) Rückgabe ───────────────────────────────────────────────
    return TREND, P_LONG, P_SHORT

@numba.njit(cache=True, fastmath=True)
def run_sim(close_p, times, trend, p_long, p_short,
            tl, ts, eL, eS,
            tpL, slL, tpS, slS,
            wait_s,
            lot_size,                 # 0.01
            contract_size,            # 100 k
            spread_price,             # 0.00001
            commission_per_lot,
            use_trend_exit=False):

    cash      = np.empty(close_p.size, np.float32)   # P/L real-€
    side_out  = np.empty(close_p.size, np.int8)
    k, pos    = 0, 0
    entry_px  = 0.0
    cooldownU = 0.0

    for i in range(close_p.size):
        price = close_p[i]
        t     = times[i]

        # ---------- EXIT ----------
        if pos != 0:
            exit_px = price - spread_price/2 if pos == +1 else price + spread_price/2

            tp_hit = (exit_px >= entry_px * (1 + tpL)) if pos == +1 else \
                     (exit_px <= entry_px * (1 - tpS))
            sl_hit = (exit_px <= entry_px * (1 - slL)) if pos == +1 else \
                     (exit_px >= entry_px * (1 + slS))
            tr_exit = (trend[i] < tl) if pos == +1 else (trend[i] > ts)

            if tp_hit or sl_hit or (use_trend_exit and tr_exit):
                pnl_rt = (exit_px - entry_px) * pos                # Preis-Δ
                cash[k] = pnl_rt * contract_size * lot_size        # €-P/L
                cash[k] -= commission_per_lot * lot_size          # Gebühren
                side_out[k] = pos
                k += 1

                pos, entry_px = 0, 0.0
                cooldownU     = t + wait_s

        # ---------- ENTRY ----------
        if pos == 0 and t >= cooldownU:
            if p_long[i] >= eL and trend[i] >= tl:
                entry_px = price + spread_price/2
                pos      = +1
            elif p_short[i] >= eS and trend[i] <= ts:
                entry_px = price - spread_price/2
                pos      = -1

    # ---------- Liquidation ----------
    if pos != 0:
        exit_px = close_p[-1] - spread_price/2 if pos == +1 else close_p[-1] + spread_price/2
        pnl_rt  = (exit_px - entry_px) * pos
        cash[k] = pnl_rt * contract_size * lot_size
        cash[k] -= commission_per_lot * lot_size
        side_out[k] = pos
        k += 1

    return cash[:k], side_out[:k]

# ═══════════════════════════════════════════════════════════════════
# 5) BACKTEST-Wrapper
# ═══════════════════════════════════════════════════════════════════
# 5) BACKTEST-Wrapper  ──────────────────────────────────────────────
def make_backtester(candles,
                    TREND_ser, P_LONG_ser, P_SHORT_ser,
                    use_trend_exit=False):

    # --- ALLES in echte NumPy-Arrays kippen ------------------------
    close_p = candles["close"].to_numpy(dtype=np.float32)

    # schneller als astype(...).to_numpy()
    times   = candles["TimeStamp"]\
                 .astype("datetime64[s]").view("int64").to_numpy()

    TREND   = np.ascontiguousarray(TREND_ser,  dtype=np.float32)
    P_LONG  = np.ascontiguousarray(P_LONG_ser, dtype=np.float32)
    P_SHORT = np.ascontiguousarray(P_SHORT_ser,dtype=np.float32)

    def backtest(tl, ts, eL, eS, tpL, slL, tpS, slS):

        pnl, side = run_sim(
            close_p, times, TREND, P_LONG, P_SHORT,     # alles Arrays
            tl, ts, eL, eS,
            tpL, slL, tpS, slS,
            WAIT_SECS,
            LOT_SIZE, CONTRACT_SIZE,
            SPREAD_PRICE, COMMISSION_PER_LOT,
            use_trend_exit
        )

        stats = {
            "long" : int((side ==  1).sum()),
            "short": int((side == -1).sum()),
            "win"  : int((pnl  >  0).sum()),
            "loss" : int((pnl  <  0).sum()),
        }
        return pnl, stats

    return backtest

# ═══════════════════════════════════════════════════════════════════
# 6) OPTUNA-Objective-Funktion

ENTRY_LONG_MIN,  ENTRY_LONG_MAX  = 0.25, 0.55
ENTRY_SHORT_MIN, ENTRY_SHORT_MAX = 0.25, 0.55

TP_LONG_MIN , TP_LONG_MAX  = 0.0010, 0.0015
TP_SHORT_MIN, TP_SHORT_MAX = 0.0010, 0.0015

SL_LONG_MIN , SL_LONG_MAX  = 0.0010, 0.0020    
SL_SHORT_MIN, SL_SHORT_MAX = 0.0010, 0.0020

TREND_LONG_MIN , TREND_LONG_MAX  = 0.65, 0.90
TREND_SHORT_MIN, TREND_SHORT_MAX = 0.55, 0.80

def make_objective(backtest):

    def objective(trial):                       # ← eine Ebene eingerückt!

        # ───────── Schwellen & Targets ────────────────────────────
        tl = trial.suggest_float("thr_long_trend",  TREND_LONG_MIN , TREND_LONG_MAX)
        ts = trial.suggest_float("thr_short_trend", TREND_SHORT_MIN, TREND_SHORT_MAX)
        if ts >= tl:
            raise TrialPruned()

        eL = trial.suggest_float("thr_entry_long",  ENTRY_LONG_MIN ,  ENTRY_LONG_MAX)
        eS = trial.suggest_float("thr_entry_short", ENTRY_SHORT_MIN, ENTRY_SHORT_MAX)

        tpL = trial.suggest_float("tp_long",  TP_LONG_MIN , TP_LONG_MAX , log=True)
        tpS = trial.suggest_float("tp_short", TP_SHORT_MIN, TP_SHORT_MAX, log=True)

        slL = trial.suggest_float(
            "sl_long",
            max(SL_LONG_MIN , tpL*0.7),
            min(SL_LONG_MAX , tpL*3.0),
        )
        slS = trial.suggest_float(
            "sl_short",
            max(SL_SHORT_MIN, tpS*0.7),
            min(SL_SHORT_MAX, tpS*3.0),
        )

        # ───────── Backtest ───────────────────────────────────────
        pnl, st = backtest(tl, ts, eL, eS, tpL, slL, tpS, slS)

        # 1) Mindest-Trades
        if st["long"] < MIN_LONG or st["short"] < MIN_SHORT:
            raise TrialPruned()

        # 2) Hard-Constraint gegen Bias
        ratio = st["long"] / (st["short"] + 1e-9)
        if ratio < 0.70 or ratio > 1.60:
            raise TrialPruned()

        # 3) Kennzahlen & Score
        sharpe = pnl.mean() / (pnl.std() + EPS)
        pf     = (pnl[pnl > 0].sum() + EPS) / (abs(pnl[pnl < 0].sum()) + EPS)

        imbalance = abs(st["long"] - st["short"]) / (st["long"] + st["short"])
        score = -(0.55 * sharpe + 0.20 * np.log(pf) - 0.25 * imbalance)
        return score

    return objective          # ← nicht vergessen!

# ───────────────────────────── 7) MAIN ─────────────────────────────
from sqlalchemy import event
import optuna
from optuna.pruners import MedianPruner
from tqdm.auto import tqdm

def main() -> None:
    # 1) Daten & Modelle
    candles = build_or_load_10m()
    TREND, P_LONG, P_SHORT = prepare_models(candles)

    backtest  = make_backtester(candles, TREND, P_LONG, P_SHORT)
    objective = make_objective(backtest)

    # 2) Storage (Timeout 180 s)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_bt_threshold_fix.db",
        engine_kwargs={"connect_args": {"timeout": 180}}
    )

    # 2a) WAL aktivieren – kompatibel für alte und neue Optuna-Version
    def _enable_wal(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.close()

    # engine kann "engine" oder "_engine" heißen
    engine = getattr(storage, "engine", getattr(storage, "_engine", None))
    if engine is not None:                      # nur wenn vorhanden
        event.listen(engine, "connect", _enable_wal)
    else:
        log.warning("Konnte WAL nicht aktivieren – kein engine-Attribut!")

    # 3) Study
    study = optuna.create_study(
        study_name="after_mapping_fix66777888146747796588577457647759754345745756856857845656876668766666699969",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=MedianPruner(n_warmup_steps=10),
    )

    # 4) Optimierung
    with tqdm(total=N_TRIALS, desc="Optuna") as bar:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            callbacks=[lambda *_: bar.update(1)],
        )

    # 5) Bestes Ergebnis auswerten
    best = study.best_trial.params
    log.info("BEST  : %s", best)
    log.info("SCORE : %.4f", study.best_value)

    pnl, st = backtest(
        best["thr_long_trend"], best["thr_short_trend"],
        best["thr_entry_long"], best["thr_entry_short"],
        best["tp_long"], best["sl_long"],
        best["tp_short"], best["sl_short"],
    )

    # --- Zusätzliche Diagnose -----------------------------------------
    avg_win  = pnl[pnl > 0].mean()
    avg_loss = -pnl[pnl < 0].mean()
    print(f"Ø-Gewinn  pro Trade : {avg_win:.6f}")
    print(f"Ø-Verlust pro Trade : {avg_loss:.6f}")
    print(f"Gewinn/Verlust-Ratio: {avg_win/avg_loss:.2f}")
    print("-" * 48)


    pf   = (pnl[pnl > 0].sum() + EPS) / (abs(pnl[pnl < 0].sum()) + EPS)
    shar = pnl.mean() / (pnl.std() + EPS)

    # 6) Reporting
    print("\n=== FINAL BACKTEST ===")
    for k, v in best.items():
        print(f"{k:18s}= {v:.6f}")
    print("-" * 48)
    print(f"Trades Long        : {st['long']}")
    print(f"Trades Short       : {st['short']}")
    tot = st['long'] + st['short']
    print(f"Total Trades       : {tot}")
    print(f"  • Win  : {st['win']} ({st['win']/tot:6.2%})")
    print(f"  • Loss : {st['loss']} ({st['loss']/tot:6.2%})")
    print(f"Profit Factor      : {pf:.4f}")
    print(f"Sharpe (Trades)    : {shar:.4f}")
    print("====================================================")

if __name__ == "__main__":
    main()
