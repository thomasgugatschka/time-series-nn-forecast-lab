from pathlib import Path
from typing import Optional

# Singleton class to manage paths and directories
# This class is used to store and retrieve paths to various directories in the project.
# It is implemented as a singleton to ensure that there is only one instance of the class throughout the project.
# This is useful because it allows us to access the same set of paths from different parts of the codebase.


class PathBuilder:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathBuilder, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return

        # If not yet initialized, proceed with initialization
        self.base_dir = Path(__file__).resolve().parent.parent
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
            'script_dir': self.base_dir / 'scripts',
            'script_train_dir': self.base_dir / 'scripts' / 'train',
            'script_eval_dir': self.base_dir / 'scripts' / 'evaluate',
            'script_preprocess_dir': self.base_dir / 'scripts' / 'preprocessing'
        }

        self.initialized = True

    def get_path(self, key: str, *args: str, dataset: Optional[str] = None, model_name: Optional[str] = None) -> Path:
        """
        Retrieve a path by key, optionally extending it with subdirectories or a dataset/model name.

        Args:
            key (str): Path key to retrieve.
            *args (str): Additional path segments.
            dataset (Optional[str]): Dataset name, for dataset-specific paths.
            model_name (Optional[str]): Model name, for model-specific paths.

        Returns:
            Path: Constructed path based on the provided key and parameters.
        """
        if key == 'training_plot_dir':
            if model_name is None:
                raise ValueError("model_name is required for training_plot_dir")
            return self.paths['result_dir'] / model_name / 'plots_training'

        elif key == 'checkpoint_dir':
            if model_name is None:
                raise ValueError("model_name is required for checkpoint_dir")
            return self.paths['model_dir'] / model_name / 'checkpoints'

        # General key with subdirectories
        elif key in self.paths:
            return self.paths[key] / Path(*args)

        else:
            raise KeyError(f"Invalid key '{key}' provided to get_path().")

    def add_path(self, key: str, value: Path):
        """
        Add or update a path dynamically.

        Args:
            key (str): Key to store the path under.
            value (Path): The Path value to store.
        """
        self.paths[key] = value


# Create an instance at the module level (global scope)
path_builder = PathBuilder()