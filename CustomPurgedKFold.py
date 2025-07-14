# CustomPurgedKFold.py
import numpy as np
from sklearn.model_selection import KFold    # ← ganz wichtig!

class CustomPurgedKFold:
    """
    Purged K-Fold mit Embargo:
    - Purging: entfernt Trainings-Indizes, die mit dem Test-Intervall überlappen
    - Embargo: blockiert pct_embargo-Anteil nach jedem Test
    """
    def __init__(self, n_splits=5, samples_info_sets=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets  # pd.Series mit Zeitstempeln
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo_size = int(n * self.pct_embargo)
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        sis = self.samples_info_sets.reset_index(drop=True)

        for train_idx, test_idx in kf.split(np.arange(n)):
            # Test-Zeitfenster
            test_times = sis.iloc[test_idx]
            start, end = test_times.min(), test_times.max()

            # Purging
            purged = [
                i for i in train_idx
                if (sis.iat[i] < start) or (sis.iat[i] > end)
            ]
            # Embargo
            emb_start = test_idx.max() + 1
            emb_end = min(n, emb_start + embargo_size)
            final_train = [
                i for i in purged
                if (i < emb_start) or (i >= emb_end)
            ]

            yield final_train, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
