# utils_ebm_safe.py
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

def _chunk_predict_ebm(model, X_df, chunk=200_000):
    """Verhindert EBM-Freeze durch Stückeln."""
    out = np.empty((len(X_df), model.n_classes_), dtype=np.float32)
    for i in range(0, len(X_df), chunk):
        sl = slice(i, i+chunk)
        out[sl] = model.predict_proba(X_df.iloc[sl])
    return out

def safe_proba(estimator, X_df):
    """Drop-in-Ersatz für predict_proba – erkennt EBMs automatisch."""
    if isinstance(estimator, ExplainableBoostingClassifier):
        return _chunk_predict_ebm(estimator, X_df)
    return estimator.predict_proba(X_df)
