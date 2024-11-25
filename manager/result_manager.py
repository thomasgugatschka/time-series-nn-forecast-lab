from manager.path_manager import path_manager
from manager.file_manager import file_manager
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path

class ResultManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResultManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance


# TODO -> maybe integrate into EXPERIMENT_MANAGER !!!!!!!

    def save_best_config(self, best_config: dict, model_name: str, dataset_name: str, ts_future, ts_past,
                         split_id="default"):

        # Convert split_id dictionary into a valid string for the file path
        if isinstance(split_id, dict):
            # Example: {'frac_train': 0.7, 'frac_val': 0.15} -> frac_train_0.7_frac_val_0.15
            split_id = "_".join(f"{key}_{value}" for key, value in split_id.items())

        """Save the best hyperparameter configuration along with additional metadata."""
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
        file_manager.save_json(best_config, best_config_file_path)

    def generate_summary_report(self, config: dict, metric: float):
        """Generate a summary report that can be saved for documentation purposes."""
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
        file_manager.save_json(summary, summary_report_path)

# Singleton instance
result_manager = ResultManager()
