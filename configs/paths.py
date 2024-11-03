from pathlib import Path

# Define the base directory as the project root
base_dir = Path(__file__).resolve().parent.parent

# Define paths relative to the project root
paths = {
    'base_dir': base_dir,
    'model_dir': base_dir / 'models',
    'data_dir': base_dir / 'data',
    'log_dir': base_dir / 'logs',
    'result_dir': base_dir / 'results',
    'experiment_dir': base_dir / 'experiments',
    'config_dir': base_dir / 'configs',
    'script_dir': base_dir / 'scripts',
    'script_train_dir': base_dir / 'scripts' / 'train',
    'script_eval_dir': base_dir / 'scripts' / 'evaluate',
    'script_preprocess_dir': base_dir / 'scripts' / 'preprocessing'
}

# Function to get a specific path
def get_path(key, *args):
    return paths[key] / Path(*args)