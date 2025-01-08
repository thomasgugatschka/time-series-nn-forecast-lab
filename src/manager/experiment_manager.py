import hashlib
import tempfile
import time

import yaml
import datetime
import json
import os
from pathlib import Path

from src.manager.gcs_manager import GCSManager

from src.manager.env_manager import EnvManager
from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager
from src.utilities.logging_utils import get_logger

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


            # Initialize and load environment variables
            self.env_manager = EnvManager()
            self.env_manager.load_env()

            self.gcs_use = self.env_manager.get_env_variable('USE_GCS', default='False', as_bool=True)
            self.gcs_bucket = self.env_manager.get_env_variable('GCS_BUCKET', default=None)


            if self.gcs_use:
                if not self.gcs_bucket:
                    raise ValueError("GCS_BUCKET must be set when USE_GCS is True")
                self.sync_manager = ExperimentSync(self.gcs_bucket)
                self.metadata_gcspath = "experiments/experiment_metadata.json"
            else:
                self.metadata_path = path_manager.get_path("experiment_dir") / "experiment_metadata.json"

            self.config_manager = config_manager
            self._experiment_name = None
            self._experiment_path = None






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
        """Generate a hash for the YAML configuration file, ignoring key order."""
        logger.info(f"Hashing configuration file: {config_path}")

        with open(config_path, 'r') as f:
            # Load YAML content as a Python dictionary
            yaml_content = yaml.safe_load(f)

        # Normalize the dictionary by sorting keys
        normalized_content = self._sort_dict(yaml_content)

        # Serialize the normalized dictionary back to a string
        normalized_yaml_str = yaml.dump(normalized_content, sort_keys=True)

        # Generate a hash of the normalized string
        return hashlib.md5(normalized_yaml_str.encode('utf-8')).hexdigest()

    # TODO -> move within _hash_config ?? since it is only needed in that function so can be placed in the function itself
    def _sort_dict(self, d):
        """Recursively sort a dictionary to ensure consistent key order."""
        if isinstance(d, dict):
            return {k: self._sort_dict(v) for k, v in sorted(d.items())}
        if isinstance(d, list):
            return [self._sort_dict(v) for v in d]
        return d



# TODO add more information into the metadata file like relevant infromation about experiment dataset, model, ts_past, ts_future etc.

    def _load_experiment_metadata(self):
        """Load experiment metadata from GCS or local file."""
        if self.gcs_use:
            return self.sync_manager.load_metadata(self.metadata_gcspath)[0]
        else:
            if file_manager.file_exists(self.metadata_path):
                return file_manager.load_json(self.metadata_path)
            return {}

    def _save_experiment_metadata(self, metadata):
        """Save experiment metadata to GCS or local file."""

        if self.gcs_use:
            logger.info(f"Saving experiment metadata to GCS: {self.metadata_gcspath}")
            self.sync_manager.save_metadata_with_retries(metadata, self.metadata_gcspath)
        else:
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

        # Save top 5 hyperparameters
        top_trials_results = all_trials_results.nsmallest(10, "val_loss")
        top_trials_results_json = top_trials_results.to_dict(orient="records")
        top_hyperparams_path = self.experiment_path / "top_hyperparameters.json"
        file_manager.save_json(top_trials_results_json, top_hyperparams_path)
        logger.info(f"Top 5 hyperparameter configurations saved to {top_hyperparams_path}")

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

    def finalize_experiment(self):
        """
        Finalize the experiment by syncing data to GCS.
        """
        if self.gcs_use:
            # IMPROVE: Should get the path including the dataset name from path_manager or complete handling
            #          is done through ExperimentManager, to have Logic to create the name within path_manager is wrong.

            dataset = self.config_manager.get_main_config().get("dataset")
            self.sync_manager.sync_experiment(self.experiment_path, f"experiments/{dataset}/{self.experiment_name}")

        logger.info(f"Experiment {self.experiment_name} finalized.")

class ExperimentSync:
    RETRY_DELAY = 1  # Seconds
    MAX_RETRIES = 5  # Number of retries for saving metadata

    def __init__(self, bucket_name: str):
        self.gcs_manager = GCSManager(bucket_name)

    def sync_experiment(self, local_experiment_path: str, gcs_experiment_path: str):
        """
        Sync a local experiment folder to GCS, uploading only new or modified files.

        Args:
            local_experiment_path (str): Path to the local experiment folder.
            gcs_experiment_path (str): Destination path in GCS.
        """
        logger.info(
            f"Start syncing local experiment to GCS: {local_experiment_path} -> gs://{self.gcs_manager.bucket_name}/{gcs_experiment_path}")

        local_path = Path(local_experiment_path)
        if not local_path.exists():
            logger.info(f"Local experiment path {local_path} does not exist. Skipping sync.")
            return

        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = Path(root) / file
                relative_path = local_file_path.relative_to(local_path)
                gcs_file_path = f"{gcs_experiment_path}/{relative_path.as_posix()}"

                # Compare file metadata before uploading
                if not self.gcs_manager.file_needs_upload(local_file_path, gcs_file_path):
                    logger.info(f"Skipping unchanged file: {local_file_path}")
                    continue

                # Upload new or modified file
                self.gcs_manager.upload_file(str(local_file_path), gcs_file_path)

        logger.info(f"Synced {local_experiment_path} to gs://{self.gcs_manager.bucket_name}/{gcs_experiment_path}")

    def load_metadata(self, metadata_path: str) -> (dict, str):
        """
        Load metadata from GCS and return its content with a version hash.

        Args:
            metadata_path (str): Path to the metadata file in GCS.

        Returns:
            tuple: (metadata content as dict, version hash as str).
        """
        try:
            data = self.gcs_manager.download_to_string(metadata_path)
            version_hash = hashlib.md5(data.encode('utf-8')).hexdigest()
            return json.loads(data), version_hash
        except FileNotFoundError:
            logger.warning(f"Metadata file {metadata_path} not found in GCS. Returning empty metadata.")
            return {}, None

    def save_metadata(self, metadata: dict, metadata_path: str, expected_version: str = None):
        """
        Save metadata to GCS, verifying version to prevent conflicts.

        Args:
            metadata (dict): Metadata content to save.
            metadata_path (str): Path to the metadata file in GCS.
            expected_version (str): Expected version hash of the metadata.

        Raises:
            RuntimeError: If the metadata version has changed during the save operation.
        """
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
            json.dump(metadata, temp_file, indent=4)
            temp_file_path = temp_file.name

        if expected_version:
            # Verify version before upload
            current_metadata, current_version = self.load_metadata(metadata_path)
            if current_version != expected_version:
                raise RuntimeError("Metadata file has been updated by another instance. Retry required.")

        self.gcs_manager.upload_file(temp_file_path, metadata_path)
        os.remove(temp_file_path)
        logger.info(f"Saved metadata to: gs://{self.gcs_manager.bucket_name}/{metadata_path}")

    def save_metadata_with_retries(self, metadata: dict, metadata_path: str):
        """
        Save metadata with retries to handle version conflicts.

        Args:
            metadata (dict): Metadata content to save.
            metadata_path (str): Path to the metadata file in GCS.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                current_metadata, current_version = self.load_metadata(metadata_path)
                # Merge or update metadata as needed
                current_metadata.update(metadata)
                self.save_metadata(current_metadata, metadata_path, expected_version=current_version)
                logger.info(f"Successfully saved metadata on attempt {attempt + 1}")
                return
            except RuntimeError as e:
                logger.warning(f"Conflict detected while saving metadata. Retrying... ({attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(self.RETRY_DELAY)

        raise RuntimeError(f"Failed to save metadata after {self.MAX_RETRIES} attempts due to version conflicts.")