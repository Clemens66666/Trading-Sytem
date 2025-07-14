import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unified_trainer.longtrend_trainer import main as long_main
from unified_trainer.entry_trainer import main as entry_main
from unified_trainer.exit_trainer import main as exit_main

def test_pipeline(tmp_path):
    cfg = yaml.safe_load(open('config.yaml'))
    cfg['raw_tick_path'] = 'RawTickDataTestData.zip'
    cfg['optuna']['n_trials'] = 1
    cfg['cv']['n_splits'] = 2
    cfg_path = tmp_path / 'cfg.yml'
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    long_main(str(cfg_path), nrows=200000)
    entry_main(str(cfg_path), nrows=100000)
    exit_main(str(cfg_path), nrows=100000)
    for key in ['longtrend','entry','exit']:
        assert os.path.exists(cfg['output_paths'][key])
