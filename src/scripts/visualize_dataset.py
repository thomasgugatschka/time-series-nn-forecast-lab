from src.manager.config_manager import ConfigManager
from src.manager.data_manager import DataManager
from src.manager.file_manager import file_manager
from src.manager.path_manager import path_manager
from src.manager.plot_manager import PlotManager
from src.utilities.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def parse_args():
    """Parse arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser(description="Visualize dataset for time series forecasting.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration file.")
    return parser.parse_args()

def main():
    # Parse configuration file
    args = parse_args()
    logger.info(f"Loading configuration from {args.config_path}")
    config_manager = ConfigManager(args.config_path)
    data_manager = DataManager(config_manager)
    plot_manager = PlotManager(config_manager, data_manager, phase="dataset_visualization")

    # Load dataset and configurations
    raw_data = data_manager.load_raw_data()
    data_manager.preprocess_data()
    cropped_data = data_manager.get_data('cropped_data')
    metadata = data_manager.get_data('metadata')

    crop_start = metadata.get("crop_start", 0)
    crop_end = metadata.get("crop_end", len(raw_data))

    logger.info(f"crop_start: {crop_start}, crop_end: {crop_end}")
    logger.info(metadata)

    # Plot 1: Full Dataset (Selected Feature and All Features)
    plot_manager.plot_full_dataset(raw_data, crop_start, crop_end, metadata)

    # Plot 2: Cropped Data (Not Normalized) with ts_past and ts_future
    plot_manager.plot_cropped_data(cropped_data, metadata)

    # Plot 3: Detailed View of Data Chunk (End with Space)
    plot_manager.plot_data_chunk(cropped_data, metadata)

    # Plot 4: Dataset with Train, Validation, and Test Splits
    plot_manager.plot_splits_with_markers(cropped_data, metadata)

    # Plot 5: Dataset with all Train Splits
    plot_manager.plot_splits_with_detailed_markers(cropped_data, metadata)

    # Save dataset configuration which where the source for the plots
    dataset = config_manager.get("dataset")
    ts_past = config_manager.get_dataset_config().get("ts_past")
    ts_future = config_manager.get_dataset_config().get("ts_future")
    output_path = path_manager.get_path("result_dir") / f"{dataset}_{ts_past}_{ts_future}" / "dataset_visualization"
    output_path.mkdir(parents=True, exist_ok=True)
    file_manager.save_yaml(config_manager.get_dataset_config(), output_path / f"config_data_{dataset}.yaml")

if __name__ == "__main__":
    main()
