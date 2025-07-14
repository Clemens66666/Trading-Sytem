#!/usr/bin/env python3
# =========================================================================
# Exit-Model Trainer v4.2  –  Stacking + FT-Transformer + 25 klassische Fts
# =========================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
import rtdl
import torch.nn as nn

import os, sys, argparse, logging, warnings, pickle, joblib, gc
from pathlib import Path
from typing  import List, Dict, Set
import numpy as np, pandas as pd

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics          import log_loss
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from lightgbm                 import LGBMClassifier
from xgboost                  import XGBClassifier
from catboost                 import CatBoostClassifier


import torch
import pytorch_lightning as pl
import rtdl
import torch.nn as nn
# ─── optionaler Transformer ───
import torch, torch.nn as nn
import pytorch_lightning as pl
import rtdl

# ═════════════ Logging ═════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ═════════════ Konstanten ═════════════
RAW_TICK_PATH    = Path("E:/RawTickData3.txt")
PARQUET_OUT      = RAW_TICK_PATH.with_suffix(".parquet")

TREND_MODEL_PKL  = Path("E:/longtrend_ensemble_dl.pkl")
ENTRY_MODEL_PKL  = Path("E:/entry_stack_fullopt1min.pkl")

PICKLE_OUT       = Path("exit_stack_v4.pkl")

TIME_COLS       = {"timestamp","time","timestmp"}
PRICE_COL       = "price"
VOLUME_ALIASES  = {"volume","tick_volume","tickvol","qty"}
EPS             = 1e-8
SEED            = 42
N_SPLITS_CV     = 5
N_TRIALS        = 80

# ═════════════ Helper ═════════════
def normalize_columns(df: pd.DataFrame)->pd.DataFrame:
    mp = {}
    for c in df.columns:
        x = c.strip().lower().replace(" ","_")
        if x in VOLUME_ALIASES: 
            x = "volume"
        mp[c] = x
    return df.rename(columns=mp)

def read_raw_csv(path:Path, chunksize:int=5_000_000):
    for ch in pd.read_csv(path, chunksize=chunksize):
        yield normalize_columns(ch)

def make_bars(df:pd.DataFrame, freq:str="5T", keep_index:bool=False):
    time_col = next((c for c in TIME_COLS if c in df.columns), None)
    if not time_col:
        raise KeyError("Keine Zeitspalte gefunden.")
    df = df.copy()
    df["__ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df.set_index("__ts", inplace=True)

    # Preisquelle
    if PRICE_COL in df.columns:
        price = df[PRICE_COL]
    elif "close" in df.columns:
        price = df["close"]
    else:
        bids = df.filter(like="bid")
        asks = df.filter(like="ask")
        price = (bids.iloc[:,0] + asks.iloc[:,0]) / 2.0

    ohlc = price.resample(freq).ohlc().rename(columns=str)
    if "volume" in df.columns:
        ohlc["volume"] = df["volume"].resample(freq).sum()
    ohlc.dropna(how="all", inplace=True)

    if keep_index:
        ohlc.index.name = "timestamp"
        return ohlc
    return ohlc.reset_index()

# ═══════════ FeatureBuilder ═══════════
class FeatureBuilder:
    PRICE_COLS   = ["open","high","low","close"]
    LAG_COLS, LAGS = ["close"], [5,12]
    VOL_WINS     = {"vol_short":12,"vol_long":72}
    SMA_WINS     = {"sma_fast":12,"sma_slow":72}
    RET_WINS     = [1,5]
    RSI_WIN      = 14
    MACD_FAST, MACD_SLOW, MACD_SIG = 12,26,9

    def __init__(self, bars:pd.DataFrame):
        self.df = bars.copy()

    @staticmethod
    def _ema(s, span):
        return s.ewm(span=span, adjust=False).mean()

    def add_basic(self):
        self.df["vwap"]       = self.df[self.PRICE_COLS].mean(1)
        self.df["bar_range"]  = self.df["high"] - self.df["low"]
        self.df["bar_return"] = self.df["close"].pct_change().fillna(0)
        return self

    def add_lags(self):
        for c in self.LAG_COLS:
            for l in self.LAGS:
                self.df[f"{c}_lag{l}"] = self.df[c].shift(l)
        return self

    def add_volume(self):
        if "volume" not in self.df.columns:
            self.df["volume"] = 0
        for n,w in self.VOL_WINS.items():
            self.df[n] = self.df["volume"].rolling(w).sum().fillna(0)
        return self

    def add_returns(self):
        for w in self.RET_WINS:
            self.df[f"ret_{w}"] = self.df["close"].pct_change(w).fillna(0)
        return self

    def add_sma(self):
        for n,w in self.SMA_WINS.items():
            self.df[n] = self.df["close"].rolling(w).mean().bfill()
        return self

    def add_rsi(self):
        d  = self.df["close"].diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        rs = up.ewm(alpha=1/self.RSI_WIN, adjust=False).mean() / \
             (dn.ewm(alpha=1/self.RSI_WIN, adjust=False).mean() + EPS)
        self.df["rsi"] = 100 - 100/(1+rs)
        return self

    def add_macd(self):
        fast = self._ema(self.df["close"], self.MACD_FAST)
        slow = self._ema(self.df["close"], self.MACD_SLOW)
        macd = fast - slow
        sig  = self._ema(macd, self.MACD_SIG)
        self.df["macd"]        = macd
        self.df["macd_signal"] = sig
        return self

    def add_ofi(self):
        if "volume" not in self.df.columns:
            self.df["volume"] = 0
        self.df["ofi_min"] = self.df["close"].diff().fillna(0) * self.df["volume"]
        return self

    def build(self):
        return (
            self.add_basic()
                .add_lags()
                .add_volume()
                .add_returns()
                .add_sma()
                .add_rsi()
                .add_macd()
                .add_ofi()
                .df.fillna(0)
        )


import torch
import torch.nn as nn
import pytorch_lightning as pl
import rtdl

import torch, gc
import numpy as np

class FTWrapper(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = rtdl.FTTransformer.make_default(
            n_num_features=input_dim,
            cat_cardinalities=None,
            last_layer_query_idx=[-1],  # Token-Pooling am letzten Token
            d_out=1,                     # Ausgabe-Dimension = 1 für Binärklassifikation
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x_num):
        logits = self.net(x_num, None)
        return logits.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
def predict_transformer(model: FTWrapper, X: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    """
    Führt Inferenz in Chargen aus, um OOM zu vermeiden.
    Gibt ein numpy-array mit allen Vorhersagen zurück.
    """
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(
                X.iloc[start : start + batch_size].values,
                dtype=torch.float32,
            ).to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(p)
            # Speicher freigeben
            del xb, p
            torch.cuda.empty_cache()
            gc.collect()
    return np.concatenate(preds, axis=0)


import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

def train_transformer(X, y, lr=1e-3, epochs=2):
    # GPU-Check und Speicher-Konfig
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        torch.backends.cuda.max_split_size_mb = 128  # Fragmentierung reduzieren
    torch.cuda.empty_cache()

    # Dataset & DataLoader
    ds = TensorDataset(
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32),
    )
    dl = DataLoader(
        ds,
        batch_size=256,       # kleinerer Batch für weniger OOM
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=4,
    )

    # Trainer-Args dynamisch zusammenbauen
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        precision=16 if use_gpu else 32,
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    if use_gpu:
        trainer_kwargs["devices"] = 1  # nur 1 GPU

    trainer = pl.Trainer(**trainer_kwargs)

    # Trainieren
    model = FTWrapper(X.shape[1], lr)
    trainer.fit(model, dl)

    # Vorhersagen in Batches (OOM vermeiden)
    model.to(device)
    model.eval()
    preds = []
    pred_dl = DataLoader(ds, batch_size=256, pin_memory=use_gpu, num_workers=2)
    with torch.no_grad():
        for xb, _ in pred_dl:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb))
            preds.append(p.cpu())
            if use_gpu:
                torch.cuda.empty_cache()
    preds = torch.cat(preds).numpy()

    return model, preds


# ═══════ robustes Feature-Mapping für Entry-Stack ═══════
def expected_entry_cols(model, all_cols: List[str]) -> List[str]:
    if isinstance(model, dict) and "feature_cols" in model:
        return list(model["feature_cols"])
    m = model["model"] if isinstance(model, dict) else model
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "estimators_"):
        for est in m.estimators_:
            if hasattr(est, "feature_names_in_"):
                return list(est.feature_names_in_)
    if hasattr(m, "n_features_in_"):
        n = int(m.n_features_in_)
        return all_cols[:n]
    return all_cols

# ═══════ Tick → Feature-Parquet + Trend/Entry-Probs ═══════
def build_full_feature_df(pq:Path) -> pd.DataFrame:
    if pq.exists():
        df = pd.read_parquet(pq)
        log.info("Parquet geladen: %s  (%dk Zeilen)", pq, len(df)//1000)
    else:
        chunks = [
            make_bars(ch, "5T", keep_index=True)
            for ch in read_raw_csv(RAW_TICK_PATH)
        ]
        df = pd.concat(chunks)
        df.reset_index().to_parquet(pq, index=False)

    fb = FeatureBuilder(df.reset_index()).build()
    fb.replace([np.inf, -np.inf], 0.0, inplace=True)

    if "index" in fb.columns:
        fb = fb.drop(columns=["index"])
    fb.set_index("timestamp", inplace=True)

    # 1-h-Trend-Features
    bars1h = make_bars(fb.reset_index(), "1H", keep_index=True)
    Xh = pd.DataFrame({
        f: (bars1h[f] if f in bars1h.columns else pd.Series(0.0, index=bars1h.index))
        for f in FEATS_LT
    })[FEATS_LT].fillna(0)

    trend_p = []
    for m in BASE_LT:
        if hasattr(m, "feature_names_in_"):
            cols = list(m.feature_names_in_)
        elif hasattr(m, "n_features_in_"):
            cols = list(Xh.columns[: int(m.n_features_in_)])
        else:
            cols = list(Xh.columns)
        trend_p.append(m.predict_proba(Xh[cols])[:,1])

    p_long = np.vstack(trend_p).mean(0)
    fb["trend_prob_long"] = pd.Series(
        META_LT.predict_proba(np.c_[1-p_long, p_long])[:,1],
        index=bars1h.index
    ).reindex(fb.index, method="ffill").values

    # Entry-Stack
    ecols = expected_entry_cols(ENTRY_CLF, list(fb.columns))
    for c in ecols:
        if c not in fb.columns:
            fb[c] = 0.0
    X_entry = fb[ecols]
    fb["p_entry_short"], fb["p_entry_long"] = ENTRY_CLF["model"].predict_proba(X_entry).T

    # Zielvariable
    fb["y_reg"] = fb["close"].shift(-1) - fb["close"]
    fb.dropna(inplace=True)
    return fb

# ═══════ Optuna Objective ═══════
def ts_split(X, n=N_SPLITS_CV):
    tss = TimeSeriesSplit(n_splits=n)
    for tr, va in tss.split(X):
        yield tr, va

def objective(trial):
    # Hyperparameter-Suche
    hp = {
        "lgb_ne":  trial.suggest_int("lgb_ne", 200, 600),
        "lgb_md":  trial.suggest_int("lgb_md", 3, 10),
        "lgb_lr":  trial.suggest_float("lgb_lr", 1e-3, 0.1, log=True),
        "xgb_ne":  trial.suggest_int("xgb_ne", 200, 600),
        "xgb_md":  trial.suggest_int("xgb_md", 3, 10),
        "xgb_lr":  trial.suggest_float("xgb_lr", 1e-3, 0.1, log=True),
        "cat_ne":  trial.suggest_int("cat_ne", 300, 800),
        "cat_md":  trial.suggest_int("cat_md", 4, 10),
        "cat_lr":  trial.suggest_float("cat_lr", 1e-3, 0.1, log=True),
        "rf_ne":   trial.suggest_int("rf_ne", 200, 600),
        "rf_md":   trial.suggest_int("rf_md", 4, 15),
        "meta_C":  trial.suggest_float("meta_C", 0.1, 10, log=True),
        "tf_lr":   trial.suggest_float("tf_lr", 1e-4, 3e-3, log=True),
    }

    # Zielvariable und Features
    y = (BARS["y_reg"] > 0).astype(int)
    X = BARS.drop(columns=list(TIME_COLS) + ["y_reg"], errors="ignore")
    oof = np.zeros((len(y), 2))

    # Zeitreihen-CV
    for fold, (tr, va) in enumerate(ts_split(X)):
        log.info("Fold %d  Train=%d  Val=%d", fold, len(tr), len(va))

        # Basis-Modelle definieren
        mdl_lgb = LGBMClassifier(
            n_estimators=hp["lgb_ne"], max_depth=hp["lgb_md"],
            learning_rate=hp["lgb_lr"], subsample=0.8,
            objective="binary", random_state=SEED, n_jobs=-1
        )
        mdl_xgb = XGBClassifier(
            n_estimators=hp["xgb_ne"], max_depth=hp["xgb_md"],
            learning_rate=hp["xgb_lr"], subsample=0.8,
            eval_metric="logloss", random_state=SEED, n_jobs=-1
        )
        mdl_cat = CatBoostClassifier(
            iterations=hp["cat_ne"], depth=hp["cat_md"],
            learning_rate=hp["cat_lr"], loss_function="Logloss",
            random_seed=SEED, verbose=False
        )
        mdl_rf = RandomForestClassifier(
            n_estimators=hp["rf_ne"], max_depth=hp["rf_md"],
            random_state=SEED, n_jobs=-1
        )

        # Trainieren der Basis-Modelle
        for m in (mdl_lgb, mdl_xgb, mdl_cat, mdl_rf):
            m.fit(X.iloc[tr], y.iloc[tr])

        # Numerische Spalten für den Transformer
        tf_cols = X.select_dtypes("number").columns.tolist()

        # Transformer auf Trainings-Fold trainieren (nur Trainings-Vorhersage, nicht fürs OOF)
        _, _ = train_transformer(
            X.iloc[tr][tf_cols], y.iloc[tr],
            lr=hp["tf_lr"], epochs=2
        )

        # Transformer endgültig trainieren für Validierungs-Vorhersage
        model, _ = train_transformer(
            X.iloc[tr][tf_cols], y.iloc[tr],
            lr=hp["tf_lr"], epochs=2
        )

        # Transformer-Vorhersage auf dem Val-Set
        tf_pred_val = predict_transformer(
            model,
            X.iloc[va][tf_cols],
            batch_size=1024
        )

        # Falls predict_transformer nur 1D zurückgibt, in 2D umformen
        if tf_pred_val.ndim == 1:
            tf_pred_val = tf_pred_val.reshape(-1, 1)

        # Stacken der Wahrscheinlichkeiten
        stack = np.column_stack([
            mdl_lgb.predict_proba(X.iloc[va])[:, 1],
            mdl_xgb.predict_proba(X.iloc[va])[:, 1],
            mdl_cat.predict_proba(X.iloc[va])[:, 1],
            mdl_rf .predict_proba(X.iloc[va])[:, 1],
            tf_pred_val.squeeze(),  # jetzt ist tf_pred_val (n_samples,1)
        ])

        # Meta-Modell trainieren & OOF-Vorhersage
        meta = LogisticRegression(C=hp["meta_C"], max_iter=1000)
        meta.fit(stack, y.iloc[va])
        oof[va] = meta.predict_proba(stack)

    # Rückgabe: OOF-Logloss
    return log_loss(y, oof, labels=[0, 1])

# ═══════ Main ═══════
def main():
    global BASE_LT, META_LT, FEATS_LT, ENTRY_CLF, BARS
    trend_obj = joblib.load(TREND_MODEL_PKL)
    entry_obj = joblib.load(ENTRY_MODEL_PKL)
    BASE_LT, META_LT = trend_obj["base_models"], trend_obj["meta"]
    FEATS_LT        = [c for c in trend_obj["feat_cols"] if c not in TIME_COLS]
    ENTRY_CLF       = entry_obj  # dict mit "model" + optional "feature_cols"

    log.info("🛠   Feature-Engineering …")
    BARS = build_full_feature_df(PARQUET_OUT)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    log.info("Best %.6f  Params=%s", study.best_value, study.best_trial.params)

if __name__ == "__main__":
    main()
