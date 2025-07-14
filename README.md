# Unified Trainer

This project contains three example training scripts and a shared helper module.
All helpers live in `unified_trainer/helpers.py` and offer utilities for
loading tick data, building features and performing time based splits.

```
unified_trainer/
    helpers.py            shared feature and utility code
    exit_trainer.py       example exit model training
    longtrend_trainer.py  example long‑trend trainer
    entry_trainer.py      example entry model trainer
config.yaml               configuration used by the trainers
```

## Running the trainers
Each trainer loads the `config.yaml` from the project root. Use the Python
interpreter of your choice:

```bash
python unified_trainer/exit_trainer.py
python unified_trainer/longtrend_trainer.py
python unified_trainer/entry_trainer.py
```


The scripts read the raw tick data specified in `raw_tick_path` and engineer
more than twenty‑five technical features through `FeatureBuilder`. Each trainer
uses a dedicated label:

* **Longtrend** – triple barrier label predicting the next 1‑3 hours.
* **Entry** – labels highs and lows within a 30‑minute window on 10‑minute bars.
* **Exit** – binary label predicting if the next 5 minute bar closes higher.

Feature windows and model hyperparameters are tuned via Optuna using a rolling
time series split. After optimisation a final model is trained on the full IS
range and stored as a pickle at the location defined under `output_paths` in
`config.yaml`.
=======
The scripts read the raw tick data specified in `raw_tick_path`, build a simple
feature frame and split it into in‑sample (IS) and two out‑of‑sample (OOS)
segments via `split_datasets`. Hyperparameters are optimised with Optuna using a
rolling time series split. After optimisation a final model is trained on the
full IS range and stored as a pickle at the location defined under
`output_paths` in `config.yaml`.


Each pickle contains:

* the trained models (`rf`, `gb` and logistic meta model)

* the preprocessor (here `None`)
=======
* a placeholder for the preprocessor (`None` in this example)

* the entire configuration
* date ranges for IS, OOS1 and OOS2

## Dataset splitting
`helpers.split_datasets(df, is_end_date, oos1_end_date)` splits any DataFrame by
its timestamp column:

1. **IS** – data up to `is_end_date`
2. **OOS1** – between `is_end_date` and `oos1_end_date`
3. **OOS2** – everything after `oos1_end_date`

The three returned DataFrames do not overlap in time.
