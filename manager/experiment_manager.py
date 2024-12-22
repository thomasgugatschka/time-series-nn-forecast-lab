import hashlib
import datetime
import json
from pathlib import Path

from manager.path_manager import path_manager
from manager.file_manager import file_manager
from utilities.logging_utils import get_logger

logger = get_logger(__name__)


class ExperimentManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_manager=None):
        """
        Initialize the ExperimentManager. Requires `config_manager` during the first instantiation.

        Args:
            config_manager: An instance of ConfigManager.
        """
        if not hasattr(self, "_initialized") or not self._initialized:
            if config_manager is None:
                raise ValueError(
                    "ExperimentManager must be initialized with a config_manager during the first instantiation."
                )

            self.config_manager = config_manager
            self._experiment_name = None
            self._experiment_path = None
            self.metadata_path = path_manager.get_path("experiment_dir") / "experiment_metadata.json"
            self._initialized = True

    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value):
        self._experiment_name = value

    @property
    def experiment_path(self):
        return self._experiment_path

    @experiment_path.setter
    def experiment_path(self, value):
        if not isinstance(value, Path):
            value = Path(value)  # Convert to Path if not already
        self._experiment_path = value


    # TODO -> add DATASET Config to the HASH ?!?!?! in case ts_future / ts_past etc. change
    def _hash_config(self, config_path):
        """Generate a hash for the configuration file."""
        logger.info(f"Hashing configuration file: {config_path}")
        with open(config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_experiment_metadata(self):
        """Load experiment metadata from the JSON file using file_manager."""
        if file_manager.file_exists(self.metadata_path):
            return file_manager.load_json(self.metadata_path)
        return {}

    def _save_experiment_metadata(self, metadata):
        """Save updated experiment metadata to the JSON file using file_manager."""
        logger.info(f"Saving experiment metadata to: {self.metadata_path}")
        file_manager.save_json(metadata, self.metadata_path)

    def manage_experiment(self):
        """
        Manage an experiment by generating a name, creating the folder, and deciding whether to resume.

        Returns:
            tuple: (experiment_name, experiment_path, resume) - name, path, and resume flag.
        """
        resume = False
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        config_path = self.config_manager.get_main_config_path()

        main_config = self.config_manager.get_main_config()
        dataset_config = self.config_manager.get_dataset_config()

        # Compute hash for the configuration file
        config_hash = self._hash_config(config_path)

        # Load existing metadata
        metadata = self._load_experiment_metadata()

        # Check if the experiment already exists
        if config_hash in metadata:
            # Resume existing experiment
            experiment_info = metadata[config_hash]
            self.experiment_name = experiment_info["experiment_name"]
            #self.experiment_path = Path(experiment_info["experiment_path"])
            resume = True
        else:
            # New experiment - Generate experiment name
            self.experiment_name = self.generate_experiment_name(main_config, dataset_config, timestamp)

        # Create experiment folder and store path
        self.experiment_path = path_manager.create_experiment_folder(
            model_type=main_config["model_type"],
            dataset_name=main_config["dataset"],
            experiment_name=self.experiment_name
        )

        path_manager.create_experiment_subfolders()

        if not resume:
            # Save metadata for the new experiment
            metadata[config_hash] = {
                "experiment_name": self.experiment_name,
                "experiment_path": str(self.experiment_path),
                "timestamp": timestamp,
            }
            self._save_experiment_metadata(metadata)


        return self.experiment_name, self.experiment_path, resume

    def generate_experiment_name(self, main_config, dataset_config, timestamp: str) -> str:
        """
        Generate a structured experiment name based on config values and timestamp.

        Args:
            main_config (dict): Main configuration.
            dataset_config (dict): Dataset-specific configuration.
            timestamp (str): Unique timestamp.

        Returns:
            str: Experiment name.
        """
        model_type = main_config.get("model_type")
        dataset = main_config.get("dataset")
        frac_train = main_config.get("data_params", {}).get("data_split", {}).get("frac_train", 0.7)
        frac_val = main_config.get("data_params", {}).get("data_split", {}).get("frac_val", 0.15)
        frac_test = main_config.get("data_params", {}).get("data_split", {}).get("frac_test", 0.15)
        ts_past = dataset_config.get("ts_past")
        ts_future = dataset_config.get("ts_future")

        # Construct name components
        components = [
            model_type, dataset, str(ts_past), str(ts_future),
            f"{frac_train:.2f}", f"{frac_val:.2f}", f"{frac_test:.2f}", timestamp
        ]

        # Join components with underscores
        return "_".join(components)

    def save_analysis_results(self, analysis):
        """
        Save the analysis object components, configuration files, and top hyperparameters.

        Args:
            analysis (tune.Analysis): The Ray Tune analysis object.

        """
        dataset_config = self.config_manager.get_dataset_config()
        main_config = self.config_manager.get_main_config()

        # Save the dataset configuration
        dataset_config_path = self.experiment_path / "dataset_config.yaml"
        file_manager.save_yaml(dataset_config, dataset_config_path)
        logger.info(f"Dataset configuration saved to {dataset_config_path}")

        # Save the main configuration
        main_config_path = self.experiment_path / "main_config.yaml"
        file_manager.save_yaml(main_config, main_config_path)
        logger.info(f"Main configuration saved to {main_config_path}")

        # Extract trial results
        all_trials_results = analysis.results_df

        # Convert DataFrame to JSON-serializable format
        all_trials_results_json = all_trials_results.to_dict(orient="records")

        # Save all hyperparameters
        all_hyperparams_path = self.experiment_path / "all_hyperparameters.json"
        file_manager.save_json(all_trials_results_json, all_hyperparams_path)
        logger.info(f"All hyperparameter configurations saved to {all_hyperparams_path}")

        # Save top 10 hyperparameters
        top_trials_results = all_trials_results.nsmallest(10, "val_loss")
        top_trials_results_json = top_trials_results.to_dict(orient="records")
        top_hyperparams_path = self.experiment_path / "top_hyperparameters.json"
        file_manager.save_json(top_trials_results_json, top_hyperparams_path)
        logger.info(f"Top 10 hyperparameter configurations saved to {top_hyperparams_path}")

        # Save best trial and best config
        best_trial = analysis.get_best_trial(metric="val_loss", mode="min").last_result
        best_trial_path = self.experiment_path / "best_trial.json"
        file_manager.save_json(best_trial, best_trial_path)
        logger.info(f"Best trial saved to {best_trial_path}")
        logger.info(f"Best Trial Metrics: {json.dumps(best_trial, indent=2)}")

        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        best_config_path = self.experiment_path / "best_config.json"
        file_manager.save_json(best_config, best_config_path)
        logger.info(f"Best config saved to {best_config_path}")
        logger.info(f"Best Hyperparameters: {json.dumps(best_config, indent=2)}")

        # Save trial-specific data
        trial_data = {
            "trial_configs": {
                trial.trial_id: trial.config
                for trial in analysis.trials
            },
            "trial_metrics": {
                trial.trial_id: trial.last_result
                for trial in analysis.trials
            }
        }

        trial_data_path = self.experiment_path / "trial_data.json"
        file_manager.save_json(trial_data, trial_data_path)
        logger.info(f"Trial-specific data saved to {trial_data_path}")

