import matplotlib.pyplot as plt

from manager.path_manager import path_manager
from manager.file_manager import file_manager

from utilities.data_utils import load_data, crop_data, calculate_features

from utilities.logging_utils import get_logger
logger = get_logger(__name__)

def main(dataset):

    # Load configuration files
    dataset_config_path = path_manager.get_path_dataset_config(dataset_name=dataset)
    dataset_config = file_manager.load_yaml(dataset_config_path)

    # Load data
    ts_past = dataset_config['ts_past']
    ts_future = dataset_config['ts_future']

    sources = dataset_config['sources']
    source_forecasted = dataset_config['source_forecasted']
    feature_forecasted = dataset_config['feature_forecasted']
    columns_to_drop = dataset_config.get('columns_to_drop')
    crop_start = dataset_config.get('crop_start')
    crop_end = dataset_config.get('crop_end')
    feature_list = dataset_config['feature_list']

    df = load_data(sources, path_manager.get_path_data(dataset_name=dataset), columns_to_drop)

    # Crop data if specified
    df = crop_data(df, crop_start, crop_end)

    # Calculate features
    remaining_features, selected_feature, num_features = calculate_features(
        feature_list, columns_to_drop, source_forecasted, feature_forecasted, sources
    )

    if crop_start is None:
        crop_start = 0

    # TODO refactor this into the plot_utils.py

    # Plot the complete dataset with ts_past and ts_future markers
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[:, selected_feature], label=remaining_features[selected_feature % len(remaining_features)],
             linewidth=2.5)
    cursor_position_ts_past = len(df) + crop_start - ts_future - ts_past
    cursor_position_ts_future = len(df) + crop_start - ts_future
    plt.axvline(x=cursor_position_ts_past, color='red', linestyle='--', label='ts_past')
    plt.axvline(x=cursor_position_ts_future, color='green', linestyle='--', label='ts_future')
    plt.ylim(df.iloc[:, selected_feature].min(), df.iloc[:, selected_feature].max())
    plt.title(
        f'{remaining_features[selected_feature % len(remaining_features)]} with ts_past and ts_future markers\nSource Forecasted: {source_forecasted}')
    plt.legend()
    plt.show()

    # Crop data to the last ts_past + ts_future rows
    crop_start = int(-(ts_past + ts_future) * 1.05)  # 5% more so cursors are visible
    df_last_chunk = df.iloc[crop_start:]
    last_position = df.index[-1]

    # Plot the last chunk of the dataset with ts_past and ts_future markers
    plt.figure(figsize=(14, 8))
    plt.plot(df_last_chunk.iloc[:, selected_feature],
             label=remaining_features[selected_feature % len(remaining_features)], linewidth=2.5)
    cursor_position_ts_past = last_position - ts_future - ts_past
    cursor_position_ts_future = last_position - ts_future
    plt.axvline(x=cursor_position_ts_past, color='red', linestyle='--', label='ts_past')
    plt.axvline(x=cursor_position_ts_future, color='green', linestyle='--', label='ts_future')
    plt.ylim(df_last_chunk.iloc[:, selected_feature].min(), df_last_chunk.iloc[:, selected_feature].max())
    plt.title(
        f'{remaining_features[selected_feature % len(remaining_features)]} with ts_past and ts_future markers (last chunk)\nSource Forecasted: {source_forecasted}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = 'weather_station'
    #dataset = 'sinusoidal'
    main(dataset)
