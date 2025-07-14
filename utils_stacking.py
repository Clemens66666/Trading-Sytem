# utils_stacking.py
# --------------------------------------------------------------------
import logging
from typing import Dict, Tuple, List
from sklearn.base import BaseEstimator

def filter_valid_estimators(models: Dict[str, BaseEstimator]) -> List[Tuple[str, BaseEstimator]]:
    """
    Entfernt alle Basismodelle, die für scikit-learn-Stacking nicht
    geeignet sind (None, kein predict_proba, NGBoost-Klassen, …).

    Gibt eine Liste [(name, est), …] zurück.
    """
    valid: List[Tuple[str, BaseEstimator]] = []

    for name, est in models.items():
        # 1) Modell existiert?
        if est is None:
            logging.info("⚠️  %s übersprungen (None-Eintrag).", name)
            continue

        # 2) Muss ein echtes scikit-Estimator-Objekt sein
        if not hasattr(est, "get_params"):
            logging.info("⚠️  %s übersprungen (kein get_params).", name)
            continue

        # 3) Stacking benötigt predict_proba
        if not hasattr(est, "predict_proba"):
            logging.info("⚠️  %s übersprungen (kein predict_proba).", name)
            continue

        # 4) NGBoost-Modelle machen im Stacker Probleme → skip
        if est.__class__.__name__.startswith(("NGB", "NGBoost")):
            logging.info("⚠️  %s übersprungen (NGBoost inkompatibel).", name)
            continue

        valid.append((name, est))

    return valid
