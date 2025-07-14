import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unified_trainer.helpers import split_datasets


def test_split_datasets():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"val": range(10)}, index=idx).reset_index().rename(columns={"index": "timestamp"})
    is_end = idx[3]
    oos1_end = idx[6]
    df_is, df_oos1, df_oos2 = split_datasets(df, is_end, oos1_end)
    assert df_is["timestamp"].max() <= is_end
    assert df_oos1["timestamp"].min() > is_end
    assert df_oos1["timestamp"].max() <= oos1_end
    assert df_oos2["timestamp"].min() > oos1_end
    assert len(set(df_is["timestamp"]).intersection(df_oos1["timestamp"])) == 0
    assert len(set(df_is["timestamp"]).intersection(df_oos2["timestamp"])) == 0
    assert len(set(df_oos1["timestamp"]).intersection(df_oos2["timestamp"])) == 0
