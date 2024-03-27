from datetime import datetime
import glob
from pathlib import Path


def create_models_dir(dataset_name: str):
    ensemble_list = glob.glob(f'models/{dataset_name}/ensemble*')
    num = 0
    for ensemble in ensemble_list:
        if int(ensemble.split('_')[-1]) > num:
            num = int(ensemble.split('_')[-1])
    model_dir = Path('.') / 'models' / dataset_name / f'ensemble_{int(num)+1}'
    model_dir.mkdir()
    return model_dir

def create_single_model_dirs(models_path: Path):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = Path('.') / 'logs' / 'gradient_tape' / current_time / 'train'
    vali_log_dir = Path('.') / 'logs' / 'gradient_tape' / current_time / '/validation'
    model_path = models_path / current_time
    model_path.mkdir()
    return model_path, train_log_dir, vali_log_dir 

