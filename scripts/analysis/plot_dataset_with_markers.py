import matplotlib.pyplot as plt
from configs.paths import get_path
from configs.logging_config import setup_logging
from utilities.data_utils import load_config, load_data, crop_data, calculate_features


def main(dataset):
    setup_logging()

    # Load configuration files
    data_config = load_config(get_path('config_dir', f'config_data_{dataset}.yaml'))

    # Load data
    ts_past = data_config['ts_past']
    ts_future = data_config['ts_future']

    sources = data_config['sources']
    source_forecasted = data_config['source_forecasted']
    feature_forecasted = data_config['feature_forecasted']
    columns_to_drop = data_config.get('columns_to_drop')
    crop_start = data_config.get('crop_start')
    crop_end = data_config.get('crop_end')
    feature_list = data_config['feature_list']

    df = load_data(sources, get_path('data_dir', dataset), columns_to_drop)

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
    #dataset = 'weather_station'
    dataset = 'sinusoidal'
    main(dataset)
