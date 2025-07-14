# trainer_globals.py — automatisch generiert

# ───────── Dateien & Pfade ─────────
DATA_RAW   = "E:/RawTickData3.txt"
MODEL_OUT  = "E:/trained_market_sentiment_ensemble_ofifinal.pkl"

# ───────── Sentiment-Parameter ─────────
SENT_PAR   = {'long_window': 400, 'r1': 0.4891927253728391, 'r2': 0.9509815990766248, 'r3': 0.01933833436183793}

# ───────── Feature-Parameter ─────────
FEAT_PAR   = {'vol_short': 19, 'vol_long': 245, 'ret1': 2, 'ret5': 7, 'rsi_w': 164, 'mf': 28, 'ms': 40, 'msig': 21, 'ofi_window': 112}

# ───────── Label-Parameter ─────────
LAB_PAR    = {'vol_window': 48, 'barrier_multiplier': 5.607210988338592, 'cusum_multiplier': 1.0341906041493873, 'horizon_seconds': 1211}

# ───────── Basis-Modelle (Einzeln) ─────────
BASE_XGB_PAR = {'learning_rate': 0.2042012200183023, 'max_depth': 3, 'n_estimators': 53, 'subsample': 0.7117567915277734, 'use_label_encoder': False, 'eval_metric': 'logloss'}
BASE_LGB_PAR = {'learning_rate': 0.1893942726269578, 'num_leaves': 76, 'n_estimators': 57, 'subsample': 0.9562830371386458}
BASE_CAT_PAR = {'learning_rate': 0.03220927391284664, 'depth': 3, 'iterations': 63}
BASE_RF_PAR  = {'n_estimators': 65, 'max_depth': 3, 'max_features': 'sqrt'}

# ───────── Gesamt Base-Model-Tuning ─────────
BASE_MODEL_TYPE = "cat"
BASE_MODEL_PAR  = {'learning_rate': 0.030165323104529306, 'depth': 9, 'iterations': 92, 'thread_count': 1, 'verbose': 0}

# ───────── Resampling-Regeln & Epsilon ─────────
BAR_RULE_MIN  = "10T"
BAR_RULE_HOUR = "4H"
EPS           = 1e-6
