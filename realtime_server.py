#!/usr/bin/env python3
# ─────────────────── server_v6.py  (Tick-Recorder Edition) ───────────────────
"""
Tick-API • Trend+Entry-Diagnose • /signal
+ Tick-Recorder: speichert Bid/Ask/Volume UND Modell-Signale in Parquet
"""
from __future__ import annotations
import os, json, logging, warnings
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, request, jsonify
import joblib

# ── Feature-Pipeline (aus deinem Backtest-Modul) ────────────────────────────
from Tick_Backtest_3 import (
    ensure_timestamp, standardize_tick_last,
    resample_to_bars, compute_ofi_min,
    MarketSentimentTransformer, FeatureAugmenter, compute_hourly_features
)

# ═══════════ 0) KONFIG ══════════════════════════════════════════════════════
LONG_MODEL  = Path(os.getenv("LONG_MODEL_PATH",  "E:/longtrend_ensemble_dl.pkl"))
ENTRY_MODEL = Path(os.getenv("ENTRY_MODEL_PATH", "E:/entry_stack_fullopt1min.pkl"))

# ▼ Trial 538 – Threshold-Konstanten ▼
THR_LONG_TREND  = 0.682414995953807
THR_SHORT_TREND = 0.5482049055309994
THR_ENTRY_LONG  = 0.44486648828925823
THR_ENTRY_SHORT = 0.2666182354375205


TP_LONG_REL , SL_LONG_REL  = 0.001017, 0.000717
TP_SHORT_REL, SL_SHORT_REL = 0.004347, 0.004418

SENT_WINDOW = 57
FEAT_PARAMS = dict(vs=5, vl=101, r1=3, r5=7,
                   rsi_w=30, mf=6, ms=27, msig=37, ofi=46)

ENTRY_FEAT_ORDER = [
    "open","high","low","close","vwap","bar_range","bar_return",
    "ofi_min","sentiment_score","vol_short","vol_long","ret_1","ret_5",
    "rsi","macd","macd_signal","trend_prob_long"
]
BAR_1M = "1T"
REC_ROOT = Path("tick_archive")          # <── Parquet-Ordner

# ═══════════ 1) LOGGING & WARNINGS ══════════════════════════════════════════
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X has feature names, but.*ExtraTreesClassifier.*"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("interpret").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ═══════════ 2) MODELLE LADEN ═══════════════════════════════════════════════
lt_dict   = joblib.load(LONG_MODEL)
BASES     = lt_dict["base_models"]
META_LT   = lt_dict["meta"]
FEAT_COLS = [c for c in lt_dict["feat_cols"] if c.lower() != "timestamp"]

entry_obj = joblib.load(ENTRY_MODEL)
ENTRY_CLF = entry_obj["model"] if isinstance(entry_obj, dict) else entry_obj

log.info("Modelle geladen – Trend-Basen: %d  Entry: %s",
         len(BASES), ENTRY_CLF.__class__.__name__)

# ═══════════ 3) TICK-PUFFER ════════════════════════════════════════════════
ticks: List[Dict[str, Any]] = []
pending_tick: Dict[str, Any] | None = None      # wird nach POST gepuffert

# ═══════════ 4) RECORDER ─ Tick + Signal → Parquet ═════════════════════════
def write_tick(tick: Dict[str, Any], signal: Dict[str, Any]) -> None:
    """Speichert einen Datensatz (Roh-Tick + Signal-Meta) in tagesweise Parquet."""
    row = {**tick, **signal}
    day_path = REC_ROOT / row["timestamp"][:10]      # YYYY-MM-DD
    day_path.mkdir(parents=True, exist_ok=True)
    pq.write_to_dataset(
        pa.Table.from_pylist([row]),
        root_path=day_path,
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore"
    )

# ═══════════ 5) FEATURE-HILFEN (1-Min-Frame) ═══════════════════════════════
def compute_1m_features(tick_df: pd.DataFrame) -> pd.DataFrame:
    mins = resample_to_bars(tick_df, BAR_1M)

    ofi  = compute_ofi_min(tick_df, FEAT_PARAMS["ofi"])
    mins = pd.merge_asof(mins.sort_values("TimeStamp"), ofi,
                         on="TimeStamp", direction="backward")

    mst   = MarketSentimentTransformer(SENT_WINDOW, .8, .1, .1)
    hours = resample_to_bars(tick_df, "1H")
    sent  = mst.transform(hours)[["TimeStamp", "sentiment_score"]]
    mins  = pd.merge_asof(mins.sort_values("TimeStamp"), sent,
                          on="TimeStamp", direction="backward")

    fe   = FeatureAugmenter(**{k:v for k,v in FEAT_PARAMS.items() if k!="ofi"})
    return fe.transform(mins).fillna(0.0)

# ganz oben (existiert schon – lassen wie es ist)
def _parse_first_json(raw: str):
    raw = raw.lstrip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
        return obj            # <─ gibt nur das erste Objekt zurück
    except Exception:
        return None

# ═══════════ 6) SIGNAL-LOGIK ═════════════════════════════════════
def predict_signal(mins: pd.DataFrame,
                   hours: pd.DataFrame) -> Dict[str, Any]:
    # ── 1) Stunden-Trend­wahrscheinlichkeit ──────────────────────
    base_p = np.stack(
        [m.predict_proba(hours[FEAT_COLS])[:, 1] for m in BASES], axis=1
    )
    p_long_trend = float(base_p[-1].mean())
    trend_p = float(
        META_LT.predict_proba([[1 - p_long_trend, p_long_trend]])[0, 1]
    )

    trend_sig = (
        "LONG"  if trend_p >= THR_LONG_TREND  else
        "SHORT" if trend_p <= THR_SHORT_TREND else
        "NEUTRAL"
    )

    # ── 2) Entry-Wahrscheinlichkeiten (Spalte-Mapping!) ──────────
    p_short, p_long = map(
        float,
        ENTRY_CLF.predict_proba(mins[ENTRY_FEAT_ORDER])[-1]  # [0]=HIGH, [1]=LOW
    )

    entry_sig = (
        "LONG"  if p_long  >= THR_ENTRY_LONG  else
        "SHORT" if p_short >= THR_ENTRY_SHORT else
        "NEUTRAL"
    )

    # ── 3) Finales Handels­signal ────────────────────────────────
    final_sig = (
        "BUY"  if (trend_sig == "LONG"  and entry_sig == "LONG")  else
        "SELL" if (trend_sig == "SHORT" and entry_sig == "SHORT") else
        "HOLD"
    )

    tp_rel, sl_rel = {
        "BUY":  (TP_LONG_REL,  SL_LONG_REL),
        "SELL": (TP_SHORT_REL, SL_SHORT_REL),
        "HOLD": (0.0, 0.0)
    }[final_sig]

    return {
        "signal":     final_sig,
        "trend_sig":  trend_sig,
        "entry_sig":  entry_sig,
        "trend_prob": round(trend_p,  6),
        "p_long":     round(p_long,  6),   # LOW-Zone-W-keit
        "p_short":    round(p_short, 6),   # HIGH-Zone-W-keit
        "tp_rel":     tp_rel,
        "sl_rel":     sl_rel,
    }

# ═══════════ 7) FLASK-ENDPOINTS ═══════════════════════════════════════════
app = Flask(__name__)

@app.route("/tick", methods=["POST"])
def tick():
    global pending_tick

    # ---- 1) Rohdaten einlesen -----------------------------------
    raw = request.get_data(as_text=True)           # Bytes → str
    data = _parse_first_json(raw)                  # <─ tolerant!
    if data is None:
        log.error("/tick bad json: %s...", raw[:120])
        return jsonify(error="invalid json"), 400

    # ---- 2) Felder extrahieren ----------------------------------
    try:
        pending_tick = {
            "timestamp": pd.to_datetime(data["timestamp"]).isoformat(),
            "bid"      : float(data.get("bid",  data.get("last"))),
            "ask"      : float(data.get("ask",  data.get("last"))),
            "volume"   : int  (data.get("volume", 1)),
        }
        ticks.append({
            "TimeStamp"  : pd.to_datetime(data["timestamp"]),
            "Tick_Last"  : pending_tick["bid"],
            "Tick_Bid"   : pending_tick["bid"],
            "Tick_Ask"   : pending_tick["ask"],
            "Tick_Volume": pending_tick["volume"],
        })
    except Exception as e:
        log.exception("/tick field error: %s", e)
        return jsonify(error="bad fields"), 400

    return jsonify(status="ok")

last_log_min: datetime | None = None

@app.route("/signal", methods=["GET"])
def signal():
    global last_log_min, pending_tick

    # ─── Sicherheitschecks ───────────────────────────────────────
    if len(ticks) < 100:
        return jsonify(signal="WAIT")

    df   = ensure_timestamp(pd.DataFrame(ticks)).pipe(standardize_tick_last)
    mins = compute_1m_features(df)
    if len(mins) < 10:
        return jsonify(signal="WAIT")

    hours = compute_hourly_features(df)
    if hours.empty:
        return jsonify(signal="WAIT")

    # ─── Unvollständige Minute entfernen ─────────────────────────
    now = datetime.utcnow().replace(second=0, microsecond=0)
    if pd.to_datetime(mins["TimeStamp"].iloc[-1]) > pd.Timestamp(now):
        mins = mins.iloc[:-1]

    # ─── Trend-Probas in Minuten-Frame forward-fillen ─────────────
    mins = mins.sort_values("TimeStamp").set_index("TimeStamp")
    base_mat   = np.stack([m.predict_proba(hours[FEAT_COLS])[:, 1] for m in BASES], axis=1)
    p_long_arr = base_mat.mean(axis=1)
    trend_probs = pd.Series(
        META_LT.predict_proba(np.column_stack([1 - p_long_arr, p_long_arr]))[:, 1],
        index=hours["TimeStamp"]
    )
    mins["trend_prob_long"] = trend_probs.reindex(mins.index, method="ffill").values
    mins = mins.reset_index()

    # ─── Signal erzeugen ──────────────────────────────────────────
    out = predict_signal(mins, hours)

    # ─── Tick + Signal → Parquet ─────────────────────────────────
    if pending_tick is not None:
        write_tick(pending_tick, out)      # Recorder
        pending_tick = None

    # ─── Periodisches Logging: genau 1× je abgeschlossener Minute ─
    if last_log_min != now:
        last_log_min = now
        log.info(
            "PERIODIC_SIGNAL | time=%s | signal=%s | trend_prob=%.4f | pL=%.4f | pS=%.4f",
            now.isoformat(timespec="seconds"),
            out["signal"],
            out["trend_prob"],
            out["p_long"],
            out["p_short"],
        )

    return jsonify(out)
#

@app.route("/reset", methods=["POST"])
def reset():
    ticks.clear()
    REC_ROOT.mkdir(exist_ok=True, parents=True)
    log.info("tick-buffer cleared via /reset")
    return jsonify(status="cleared")

# ═══════════ 8) START ════════════════════════════════════════════
if __name__ == "__main__":
    REC_ROOT.mkdir(exist_ok=True, parents=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)
