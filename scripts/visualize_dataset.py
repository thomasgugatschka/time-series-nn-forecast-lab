import matplotlib.pyplot as plt
from manager.config_manager import ConfigManager
from manager.data_manager import DataManager
from manager.path_manager import path_manager
from utilities.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def parse_args():
    """Parse arguments for the script."""
    import argparse
    parser = argparse.ArgumentParser(description="Visualize dataset for time series forecasting.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration file.")
    return parser.parse_args()


# TODO -> store PLOTS in dataset folder????? or in experiment folder?????
# TODO -> move plots to utils!!!!!! so same settings config is applied for consistency!!!!!

def main():
    # Parse configuration file
    args = parse_args()
    logger.info(f"Loading configuration from {args.config_path}")
    config_manager = ConfigManager(args.config_path)
    dataset_config = config_manager.get_dataset_config()
    main_config = config_manager.get_main_config()

    data_manager = DataManager(config_manager)

    # Load dataset and configurations
    raw_data = data_manager.load_raw_data()
    data_manager.preprocess_data()
    cropped_data = data_manager.get_data('cropped_data')
    metadata = data_manager.get_data('metadata')


    # Extract configurations
    dataset_name = main_config.get('dataset', 'Unknown Dataset')
    ts_past = dataset_config['ts_past']
    ts_future = dataset_config['ts_future']
    crop_start = dataset_config.get('crop_start', 0)
    crop_end = dataset_config.get('crop_end', len(raw_data))
    selected_feature = metadata['selected_feature']
    sources = dataset_config.get('sources', [])
    feature_list = dataset_config['feature_list']

    combined_feature_list = metadata['combined_feature_list']

    # Ensure crop_start and crop_end are valid
    crop_start = 0 if crop_start is None else crop_start
    crop_end = len(raw_data) if crop_end is None else crop_end


    # Plot 1: Full Dataset (Selected Feature and All Features)
    plot_full_dataset(raw_data, dataset_name, crop_start, crop_end, selected_feature, combined_feature_list, sources)

    # Plot 2: Cropped Data (Not Normalized) with ts_past and ts_future
    plot_cropped_data(cropped_data, dataset_name, ts_past, ts_future, selected_feature, combined_feature_list)

    # Plot 3: Detailed View of Data Chunk (End with Space)
    plot_data_chunk(cropped_data, dataset_name, ts_past, ts_future, selected_feature, combined_feature_list)


def plot_full_dataset(df, dataset_name, crop_start, crop_end, selected_feature, combined_feature_list, sources):
    """Plot the full dataset with crop markers."""


    logger.info("Plotting full dataset.")

    # Plot only the selected feature
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[:, selected_feature],
             label=f"Selected Feature: {combined_feature_list[selected_feature]}", linewidth=2.5)
    plt.axvline(crop_start, color='red', linestyle='--', label="Crop Start")
    plt.axvline(crop_end, color='green', linestyle='--', label="Crop End")
    plt.title(f"Full Dataset ({dataset_name}) - Selected Feature\nCrop Start: {crop_start}, Crop End: {crop_end}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.annotate(f"Data Source: {sources}\nDatapoints: {len(df)}", xy=(0.02, 0.95), xycoords="axes fraction")
    plt.show()

    # Plot all features
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(df.columns):
        plt.plot(df.iloc[:, i], label=f"{combined_feature_list[i]}")
    plt.axvline(crop_start, color='red', linestyle='--', label="Crop Start")
    plt.axvline(crop_end, color='green', linestyle='--', label="Crop End")
    plt.title(f"Full Dataset ({dataset_name}) - All Features\nCrop Start: {crop_start}, Crop End: {crop_end}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_cropped_data(df, dataset_name, ts_past, ts_future, selected_feature, combined_feature_list):
    """Plot cropped data with ts_past and ts_future markers."""
    logger.info("Plotting cropped data with ts_past and ts_future markers.")
    plt.figure(figsize=(14, 8))

    plt.plot(df.iloc[:, selected_feature], label=f"Selected Feature: {combined_feature_list[selected_feature]}", linewidth=2.5)
    last_index = df.index[-1]
    cursor_position_ts_past = last_index - ts_future - ts_past
    cursor_position_ts_future = last_index - ts_future

    plt.axvline(cursor_position_ts_past, color='red', linestyle='--', label="ts_past")
    plt.axvline(cursor_position_ts_future, color='green', linestyle='--', label="ts_future")

    plt.title(f"Cropped Dataset ({dataset_name}) - Selected Feature\nData Points: {len(df)}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_data_chunk(df, dataset_name, ts_past, ts_future, selected_feature, combined_feature_list):
    """Plot a chunk of the data with ts_past and ts_future markers."""
    logger.info("Plotting a detailed chunk of data.")
    chunk_size = int((ts_past + ts_future) * 1.05)  # Include some space around ts_past and ts_future
    start = max(0, len(df) - chunk_size)
    df_chunk = df.iloc[start:]

    plt.figure(figsize=(14, 8))
    plt.plot(df_chunk.iloc[:, selected_feature], label=f"Selected Feature: {combined_feature_list[selected_feature]}",
             linewidth=2.5)
    last_index = df_chunk.index[-1]
    cursor_position_ts_past = last_index - ts_future - ts_past
    cursor_position_ts_future = last_index - ts_future

    plt.axvline(cursor_position_ts_past, color='red', linestyle='--', label="ts_past")
    plt.axvline(cursor_position_ts_future, color='green', linestyle='--', label="ts_future")

    plt.title(f"Detailed Data Chunk ({dataset_name}) - Selected Feature\nData Points: {len(df_chunk)}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
