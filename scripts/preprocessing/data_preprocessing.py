import os
import logging
import numpy as np
import pandas as pd
import yaml
from configs.paths import get_path
from configs.logging_config import setup_logging
from utilities.data_processing_utils import z_score_normalize

setup_logging()

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    # Splits a multivariate input sequence into past and future samples
    # input_sequences   : multivariate sequence inputed (x)
    # output_sequence   : multivariate sequence outputed (y)
    # n_steps_in        : number of time steps to look back
    # n_steps_out       : number of time steps to look in the future

    x, y = list(), list() # instantiate x and y

    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


def load_and_preprocess_data(config_data, config_train, dataset_name):
    # Load information from config_data
    sources = config_data['sources']
    source_forecasted = config_data['source_forecasted']
    feature_list = config_data['feature_list']
    feature_forecasted = config_data['feature_forecasted']
    columns_to_drop = config_data.get('columns_to_drop', [])
    crop_start = config_data.get('crop_start')
    crop_end = config_data.get('crop_end')

    # Load information from config_train
    ts_past = config_train['ts_past']
    ts_future = config_train['ts_future']

    # Extract dataset name from config_data_file
    #dataset_name = os.path.basename(config_data_file).replace('config_data_', '').replace('.yaml', '')
    data_dir = get_path('data_dir', dataset_name)
    logging.info(f"Loading data from {data_dir}")

    # Import time-series from specified sources
    df_list = []
    for source in sources:
        target_file = os.path.join(data_dir, f"{source}.csv")
        logging.info(f"Loading data from {target_file}")
        df = pd.read_csv(target_file, encoding='utf-8')  # Specify UTF-8 encoding
        logging.info(f"Data shape before dropping columns: {df.shape}")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.info(f"Data shape after dropping columns: {df.shape}")
        df_list.append(df)
        logging.info(f"Appended data from {source}.csv")

    # Crop data if specified in config_data to reduce computation time
    df = pd.concat(df_list, axis=1)
    if crop_start is not None and crop_end is not None:
        df = df.iloc[crop_start:crop_end, :]

    # Preprocessing data with z-score normalization
    df, z_mean, z_std = z_score_normalize(df)

    # Calculate selected feature and number of features considering dropped columns
    if columns_to_drop is None:
        columns_to_drop = []

    remaining_features = [feature for feature in feature_list if feature not in columns_to_drop]
    selected_feature = len(remaining_features) * source_forecasted + feature_forecasted
    num_features = len(sources) * len(remaining_features)

    # Feed context and prediction results to the model (adding another dimension)
    # The split_sequences function splits the multivariate time series data into
    #   input (x_arr) and output (y_arr) sequences for supervised learning.
    x_arr, y_arr = split_sequences(df, df.iloc[:, selected_feature], ts_past, ts_future)

    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"x_arr shape: {x_arr.shape} - Input sequences for the model")
    logging.info(f"y_arr shape: {y_arr.shape} - Output sequences for the model")
    logging.info(f"z_mean: {z_mean} - Mean values used for z-score normalization")
    logging.info(f"z_std: {z_std} - Standard deviation values used for z-score normalization")
    logging.info(f"selected_feature: {selected_feature} - Index of the selected feature for forecasting")
    logging.info(f"num_features: {num_features} - Total number of features after preprocessing")

    return df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features


def main(config_data_file, config_train_file, dataset_name):
    with open(config_data_file, 'r') as file:
        config_data = yaml.safe_load(file)
    with open(config_train_file, 'r') as file:
        config_train = yaml.safe_load(file)

    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = load_and_preprocess_data(config_data, config_train, dataset_name)
    print("Data loaded and preprocessed successfully.")
    print(f"DataFrame shape: {df.shape}")
    print(f"x_arr shape: {x_arr.shape}")
    print(f"y_arr shape: {y_arr.shape}")

if __name__ == "__main__":
    config_data_file = get_path('config_dir', 'config_data_sinusoidal.yaml')
    config_train_file = get_path('config_dir', 'config_train_rnn_sinusoidal.yaml')
    main(config_data_file, config_train_file, 'sinusoidal')