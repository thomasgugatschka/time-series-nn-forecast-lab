from pathlib import Path
from typing import Optional, Dict
from google.cloud import storage
import os
from src.manager.env_manager import EnvManager

# Initialize logging
from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)


class PathManager:
    """
    Singleton class to manage paths and directories in the project.
    The root directory (PROJECT_BASE_DIR) is dynamically resolved based on the .env file.
    """

    _instance = None

    DATASET_CONFIG_FILE_TEMPLATE = "config_data_{dataset_name}.yaml"

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return


        # Initialize and load environment variables
        self.env_manager = EnvManager()
        self.env_manager.load_env()


        # Retrieve PROJECT_BASE_DIR, ensuring the default value is a string
        default_base_dir = str(Path(__file__).resolve().parent.parent)
        self.base_dir = Path(self.env_manager.get_env_variable('PROJECT_BASE_DIR', default_base_dir)).resolve()

        logger.info(f"Project Base Directory: {self.base_dir}")


        self.paths = {
            'base_dir': self.base_dir,
            'data_dir': self.base_dir / 'data',
            'experiment_dir': self.base_dir / 'experiments',
            'log_dir': self.base_dir / 'logs',
            'config_dir': self.base_dir / 'configs',
            'dataset_config_dir': self.base_dir / 'configs' / 'dataset_configs',
        }


        self._ensure_directories(self.paths.values())

        # Placeholder for dynamically added paths
        self.dynamic_paths: Dict[str, Path] = {}
        self.initialized = True


    @staticmethod
    def _ensure_directories(directories):
        """
        Ensure all directories exist.
        """
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)

    def get_path(self, path_name: str) -> Path:
        """
        Retrieve a core or dynamically added path by its name.
        Args:
            path_name (str): The name of the path.
        Returns:
            Path: Resolved path.
        Raises:
            KeyError: If the path_name is not defined.
        """
        if path_name in self.paths:
            return self.paths[path_name]
        if path_name in self.dynamic_paths:
            return self.dynamic_paths[path_name]
        raise KeyError(f"Path '{path_name}' is not defined.")

    def add_path(self, path_name: str, path_value: Path):
        """
        Add or update a dynamically defined path.

        Args:
            path_name (str): Name of the path.
            path_value (Path): Path to associate with the name.
        """
        self.dynamic_paths[path_name] = path_value.resolve()
        path_value.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dynamically added path: {path_name} -> {path_value}")

    def get_path_config(self) -> Path:
        """
        Retrieve the path to the configuration directory.
        """
        return self.get_path("config_dir")

    def get_path_data(self, dataset_name: Optional[str] = None) -> Path:
        """
        Retrieve the path to the data directory, optionally for a specific dataset.
        Args:
            dataset_name (str, optional): The name of the dataset.
        Returns:
            Path: Path to the data directory or dataset directory.
        """
        data_path = self.get_path("data_dir")
        return data_path / dataset_name if dataset_name else data_path

    def get_path_dataset_config(self, dataset_name: Optional[str] = None) -> Path:
        """
        Retrieve the path to the dataset configuration directory or a specific dataset configuration file.
        Args:
            dataset_name (str, optional): Name of the dataset.
        Returns:
            Path: Path to the dataset configuration directory or file.
        """
        dataset_config_dir = self.get_path("dataset_config_dir")
        if dataset_name:
            return dataset_config_dir / self.DATASET_CONFIG_FILE_TEMPLATE.format(dataset_name=dataset_name)
        return dataset_config_dir

    def get_path_dataset_plots(self, dataset_name: str) -> Path:
        dataset_plots_dir = self.get_path("experiment_dir") / "dataset_plots" / dataset_name
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)
        return dataset_plots_dir

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
        # TODO - maybe adapt -> only experiment name
        # TODO -> experiment_name might be adapted / increased with more data ????
        experiment_path = self.get_path("experiment_dir") / model_type / dataset_name / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Add experiment path dynamically for later use
        # self.add_path(f"experiment:{experiment_name}", experiment_path) # Relevant in case of multiple experiments
        self.add_path(f"experiment", experiment_path)
        return experiment_path

    def create_experiment_subfolders(self, subfolders: Optional[list] = None) -> Dict[str, Path]:
        """
        Create subfolders within an experiment folder dynamically.

        Args:
            subfolders (list, optional): List of subfolder names to create. Defaults to standard subfolders.

        Returns:
            dict: Dictionary containing created subfolder paths.
        """
        if subfolders is None:
            subfolders = ["ray_results", "tensorboard", "trained_models", "evaluation_results",
                          "plots_tuning", "plots_training", "plots_evaluation", "plots_hyperparameters"]

        #experiment_path = self.get_path(f"experiment:{experiment_name}")
        experiment_path = self.get_path(f"experiment")
        subfolder_paths = {}

        for subfolder in subfolders:
            folder_path = experiment_path / subfolder
            folder_path.mkdir(parents=True, exist_ok=True)
            #self.add_path(f"{experiment_name}:{subfolder}", folder_path) # Relevant in case of multiple experiments
            self.add_path(f"experiment_{subfolder}", folder_path)
            subfolder_paths[subfolder] = folder_path

        return subfolder_paths



# Create an instance at the module level (global scope)
path_manager = PathManager()