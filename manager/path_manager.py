from pathlib import Path
from typing import Optional

# TODO nice Description of the class
# TODO which pattern was used
# Singleton class to manage paths and directories
# This class is used to store and retrieve paths to various directories in the project.
# It is implemented as a singleton to ensure that there is only one instance of the class throughout the project.
# This is useful because it allows us to access the same set of paths from different parts of the codebase.


class PathManager:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return

        # If not yet initialized, proceed with initialization
        self.base_dir = Path(__file__).resolve().parent.parent

        # TODO - CLEANUP ALL DEFINITIONS HERE WHICH ARE NOT NEEDED / REDUNDANT
        self.paths = {
            'base_dir': self.base_dir,
            'model_dir': self.base_dir / 'models',
            'data_dir': self.base_dir / 'data',
            'log_dir': self.base_dir / 'logs',
            'result_dir': self.base_dir / 'results',
            'experiment_dir': self.base_dir / 'experiments',
            'config_dir': self.base_dir / 'configs',
            'dataset_config_dir': self.base_dir / 'configs' / 'dataset_configs',
            'best_hyperparameters_dir': self.base_dir / 'configs' / 'best_hyperparameters',
            'ray_results_dir': self.base_dir / 'ray_results',
            'script_dir': self.base_dir / 'scripts',
            'script_train_dir': self.base_dir / 'scripts' / 'train',
            'script_eval_dir': self.base_dir / 'scripts' / 'evaluate',
            'script_preprocess_dir': self.base_dir / 'scripts' / 'preprocessing',
            'tensorboard_log_dir': self.base_dir / 'logs' / 'tensorboard'
        }

        self.initialized = True

    def get_path_base(self) -> Path:
        return self.paths['base_dir']

    def get_path_model(self, model_name: Optional[str] = None) -> Path:
        if model_name:
            return self.paths['model_dir'] / model_name
        return self.paths['model_dir']

    def get_path_data(self, dataset_name: Optional[str] = None) -> Path:
        if dataset_name:
            return self.paths['data_dir'] / dataset_name
        return self.paths['data_dir']

    def get_path_log(self) -> Path:
        return self.paths['log_dir']

    def get_tensorboard_log_dir(self, model_name: Optional[str] = None, experiment_name: Optional[str] = None) -> Path:
        log_path = self.paths['tensorboard_log_dir']
        if model_name and experiment_name:
            return log_path / model_name / experiment_name
        return log_path

    def get_path_result(self, model_name: Optional[str] = None) -> Path:
        if model_name:
            return self.paths['result_dir'] / model_name
        return self.paths['result_dir']

    def get_path_experiment(self) -> Path:
        return self.paths['experiment_dir']


    def get_ray_results_dir(self) -> Path:
        return self.paths['ray_results_dir']


    def get_path_config(self) -> Path:
        return self.paths['config_dir']

    def get_path_dataset_config(self, dataset_name: Optional[str] = None) -> Path:
        if dataset_name:
            return self.paths['dataset_config_dir'] / f'config_data_{dataset_name}.yaml'
        return self.paths['dataset_config_dir']

    def get_path_training_plot(self, model_name: str) -> Path:
        if not model_name:
            raise ValueError("Model name is required for training plot path.")
        return self.paths['result_dir'] / model_name / 'plots_training'

    def get_path_checkpoint(self, model_name: str) -> Path:
        if not model_name:
            raise ValueError("Model name is required for checkpoint path.")
        return self.paths['model_dir'] / model_name / 'checkpoints'

    def get_path_hyperparameter_config(self) -> Path:
        return self.paths['config_dir'] / 'model_hyperparameter_config.json'

    def get_path_best_hyperparameters(self, model_name: str, dataset_name: str, split_id: str,
                                      ts_future: Optional[int] = None, ts_past: Optional[int] = None) -> Path:
        """
        Generate a unique path for storing the best hyperparameters, based on model type, dataset, and parameters.

        Args:
            model_name (str): Name of the model.
            dataset_name (str): Name of the dataset.
            split_id (str): Identifier for the data split (e.g., 'default', 'split1').
            ts_future (Optional[int]): Future time steps.
            ts_past (Optional[int]): Past time steps.

        Returns:
            Path: Path where the best hyperparameters JSON file should be saved.
        """
        # Create a directory structure based on model, dataset, split, ts_future, and ts_past parameters at the same level
        hyperparameters_dir = self.paths['best_hyperparameters_dir'] / model_name / dataset_name

        # Create folder name for the split and optional ts_future and ts_past
        split_folder_name = f'split_{split_id}'
        if ts_future is not None:
            split_folder_name += f'_ts_f_{ts_future}'
        if ts_past is not None:
            split_folder_name += f'_ts_p_{ts_past}'

        hyperparameters_dir = hyperparameters_dir / split_folder_name

        # Return the final path to the JSON file
        return hyperparameters_dir / 'best_hyperparameters.json'


# Create an instance at the module level (global scope)
path_manager = PathManager()