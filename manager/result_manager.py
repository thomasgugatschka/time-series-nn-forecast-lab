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

    def save_best_config(self, best_config: dict, model_name: str, dataset_name: str, ts_future, ts_past,
                         split_id="default"):
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

    def log_to_tensorboard(self, model_type: str, config: dict, metric_value: float):
        """Log hyperparameters and the metric value to TensorBoard."""
        log_dir = path_manager.get_path_log() / "tensorboard_logs" / model_type
        writer = SummaryWriter(log_dir=str(log_dir))

        # Use `add_hparams()` to log hyperparameters and final metrics to TensorBoard
        writer.add_hparams(config, {"hparam/val_loss": metric_value})
        writer.close()

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
