import hashlib
import datetime
from manager.path_manager import path_manager
from manager.file_manager import file_manager
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class ExperimentManager:
    def __init__(self):
        # Set the path for experiment metadata using path_manager
        self.metadata_path = path_manager.get_path_experiment() / 'experiment_metadata.json'

    def hash_config(self, config_path):
        """Generate a hash for the configuration file."""
        logger.info(f"Hashing configuration file: {config_path}")
        with open(config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_experiment_metadata(self):
        """Load experiment metadata from the JSON file using file_manager."""
        if file_manager.file_exists(self.metadata_path):
            return file_manager.load_json(self.metadata_path)
        return {}

    def save_experiment_metadata(self, metadata):
        """Save updated experiment metadata to the JSON file using file_manager."""
        logger.info(f"Saving experiment metadata to: {self.metadata_path}")
        file_manager.save_json(metadata, self.metadata_path)

    def manage_experiment(self, model_type, dataset, ts_future, ts_past, config_path):
        """
        Check if an experiment should resume or create a new one.

        Args:
            model_type (str): The model type.
            dataset (str): The dataset name.
            ts_future (int): Number of future time steps.
            ts_past (int): Number of past time steps.
            config_path (str): Path to the configuration file.

        Returns:
            tuple: (experiment_name, resume) - experiment_name is a string, resume is a boolean.
        """
        # Step 1: Compute Hash of the Config File
        config_hash = self.hash_config(config_path)

        # Step 2: Retrieve Existing Metadata
        metadata = self.get_experiment_metadata()

        # Step 3: Determine if Experiment Resumes or Starts New
        if config_hash in metadata:
            experiment_info = metadata[config_hash]
            resume = True
            experiment_name = experiment_info['experiment_name']
            logger.info(f"Resuming experiment with hash: {config_hash}")
        else:
            resume = False
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            experiment_name = f"{model_type}_{dataset}_{ts_future}_{ts_past}_{timestamp}"
            metadata[config_hash] = {"experiment_name": experiment_name, "timestamp": timestamp}
            logger.info(f"Starting new experiment: {experiment_name}")
            self.save_experiment_metadata(metadata)

        return experiment_name, resume

    def save_best_config(self, best_config: dict, model_name: str, dataset_name: str, ts_future, ts_past, split_id="default"):
        """
        Save the best hyperparameter configuration along with additional metadata.

        Args:
            best_config (dict): The best configuration.
            model_name (str): The model type name.
            dataset_name (str): The dataset name.
            ts_future (int): Number of future time steps.
            ts_past (int): Number of past time steps.
            split_id (str or dict): Split information (can be a dictionary).
        """
        # Convert split_id dictionary into a valid string for the file path
        if isinstance(split_id, dict):
            split_id = "_".join(f"{key}_{value}" for key, value in split_id.items())

        # Generate a unique path for best hyperparameters based on different factors
        best_config_file_path = path_manager.get_path_best_hyperparameters(
            model_name=model_name,
            dataset_name=dataset_name,
            split_id=split_id,
            ts_future=ts_future,
            ts_past=ts_past
        )

        # Ensure the directory exists before saving
        file_manager.create_directory(best_config_file_path.parent)

        # Save the extended best configuration
        logger.info(f"Saving best configuration to: {best_config_file_path}")
        file_manager.save_json(best_config, best_config_file_path)

    def generate_summary_report(self, config: dict, metric: float):
        """
        Generate a summary report that can be saved for documentation purposes.

        Args:
            config (dict): Configuration of the experiment.
            metric (float): Best metric value.
        """
        # Prepare a unique path to store the summary report
        summary_report_path = path_manager.get_path_result() / "summary_reports" / f"{config['model_type']}_{config['dataset']}_summary.json"

        # Ensure the directory exists before saving
        file_manager.create_directory(summary_report_path.parent)

        # Create a summary with all relevant information
        summary = {
            "tuning_timestamp": datetime.datetime.now().isoformat(),
            "model_type": config["model_type"],
            "dataset": config["dataset"],
            "best_metric_value": metric,
            **config
        }
        logger.info(f"Saving summary report to: {summary_report_path}")
        file_manager.save_json(summary, summary_report_path)


# Example of creating an instance of the ExperimentManager
experiment_manager = ExperimentManager()
