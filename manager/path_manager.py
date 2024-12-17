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
            'data_pipeline_dir': self.base_dir / 'data_pipeline',
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

    def get_path_data_pipeline(self) -> Path:
        return self.paths['data_pipeline_dir']

    def get_path_log(self) -> Path:
        return self.paths['log_dir']

    def get_path_tensorboard(self, experiment_path) -> Path:
        """
        Get the TensorBoard log directory for a given experiment path.

        Args:
            experiment_path (Path or str): Base path for the experiment.

        Returns:
            Path: The TensorBoard log directory.
        """
        # Ensure experiment_path is a Path object
        experiment_path = Path(experiment_path)
        tensorboard_log_path = experiment_path / "tensorboard"
        return tensorboard_log_path

    def get_path_result(self, model_name: Optional[str] = None) -> Path:
        if model_name:
            return self.paths['result_dir'] / model_name
        return self.paths['result_dir']

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

    def get_path_hyperparameter_search_space(self) -> Path:
        return self.paths['config_dir'] / 'model_hyperparameter_search_space.json'


    def get_path_experiment(self) -> Path:
        return self.paths['experiment_dir']

    def get_path_experiment_metadata(self) -> Path:
        return self.get_path_experiment() / 'experiment_metadata.json'

    def create_experiment_folder(self, model_type: str, dataset_name: str, experiment_name: str) -> Path:
        """
        Create a structured folder path for an experiment.

        Args:
            model_type (str): Type of the model (e.g., RNN, Transformer).
            dataset_name (str): Name of the dataset.
            experiment_name (str): Unique name for the experiment.

        Returns:
            Path: The full path to the experiment folder.
        """
        # Construct the folder structure
        experiment_path = self.get_path_experiment() / model_type / dataset_name / experiment_name

        # Ensure the directory exists
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Return the full experiment path
        return experiment_path

    def create_experiment_subfolders(self, experiment_path: Path) -> dict:
        """
        Create required subfolders ('ray_results' and 'tensorboard') within an experiment folder.

        Args:
            experiment_path (Path): Base path for the experiment.

        Returns:
            dict: Dictionary containing paths to the created subfolders.
                  Example: {"ray_results": Path, "tensorboard": Path}
        """
        # Define subfolder names
        subfolders = ["ray_results", "tensorboard", "trained_models", "evaluation_results"]

        # Create subfolders and store their paths in a dictionary
        subfolder_paths = {}
        for subfolder in subfolders:
            folder_path = experiment_path / subfolder
            folder_path.mkdir(parents=True, exist_ok=True)
            subfolder_paths[subfolder] = folder_path

        return subfolder_paths



# Create an instance at the module level (global scope)
path_manager = PathManager()