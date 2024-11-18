import os
import logging
import pandas as pd
import numpy as np
import yaml
import torch
from configs.paths import get_path

# Load configuration function
def load_config(config_file):
    logging.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded successfully from {config_file}")
    return config

# Load data function
def load_data(sources, data_dir, columns_to_drop):
    df_list = []
    for source in sources:
        target_file = os.path.join(data_dir, f"{source}.csv")
        logging.info(f"Loading data from {target_file}")
        df = pd.read_csv(target_file, encoding='utf-8')
        logging.info(f"Data shape before dropping columns: {df.shape}")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.info(f"Data shape after dropping columns: {df.shape}")
        df_list.append(df)
        logging.info(f"Appended data from {source}.csv")
    return pd.concat(df_list, axis=1)

# Crop data function
def crop_data(df, crop_start, crop_end):
    if crop_start is not None and crop_end is not None:
        df = df.iloc[crop_start:crop_end, :]
    elif crop_start is not None:
        df = df.iloc[crop_start:, :]
    elif crop_end is not None:
        df = df.iloc[:crop_end, :]
    return df

# Z-score normalization function
def z_score_normalize(df):
    z_mean = df.mean()
    z_std = df.std()
    normalized_df = (df - z_mean) / z_std
    return normalized_df, z_mean, z_std

# Z-score denormalization function
def z_score_denormalize(normalized_df, z_mean, z_std):
    denormalized_df = (normalized_df * z_std) + z_mean
    return denormalized_df

# Split sequences function
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    x, y = list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences): break
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix - 1:out_end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# Split and convert data function
def split_and_convert_data(x_arr, y_arr, frac_train=0.7, frac_val=0.15):
    num_samples = x_arr.shape[0]
    num_train = int(frac_train * num_samples)
    num_val = int(frac_val * num_samples)
    num_test = num_samples - num_train - num_val

    # Split the data into train/validate/test
    x_arr_train = x_arr[:num_train, :, :]
    x_arr_val = x_arr[num_train:num_train + num_val, :, :]
    x_arr_test = x_arr[num_train + num_val:, :, :]
    y_arr_train = y_arr[:num_train, :]
    y_arr_val = y_arr[num_train:num_train + num_val, :]
    y_arr_test = y_arr[num_train + num_val:, :]
    logging.info(f"\nx_arr shapes:\n{x_arr_train.shape}\n{x_arr_val.shape}\n{x_arr_test.shape}")
    logging.info(f"\ny_arr shapes:\n{y_arr_train.shape}\n{y_arr_val.shape}\n{y_arr_test.shape}")

    # Convert to tensors
    x_train = torch.tensor(x_arr_train, dtype=torch.float32)
    x_val = torch.tensor(x_arr_val, dtype=torch.float32)
    x_test = torch.tensor(x_arr_test, dtype=torch.float32)
    y_train = torch.tensor(y_arr_train, dtype=torch.float32)
    y_val = torch.tensor(y_arr_val, dtype=torch.float32)
    y_test = torch.tensor(y_arr_test, dtype=torch.float32)
    logging.info(f"\nTensor shapes (x):\n{x_train.shape}\n{x_val.shape}\n{x_test.shape}")
    logging.info(f"\nTensor shapes (y):\n{y_train.shape}\n{y_val.shape}\n{y_test.shape}\n")

    logging.info(f"""
    Data split and conversion completed:
    - x_train, x_val, x_test: These are the input features for the training, validation, and test sets, respectively. 
      They are PyTorch tensors of shape (num_samples, num_timesteps, num_features).
      Shapes:
      - x_train: {x_train.shape}
      - x_val: {x_val.shape}
      - x_test: {x_test.shape}

    - y_train, y_val, y_test: These are the target values for the training, validation, and test sets, respectively. 
      They are PyTorch tensors of shape (num_samples, num_targets).
      Shapes:
      - y_train: {y_train.shape}
      - y_val: {y_val.shape}
      - y_test: {y_test.shape}

    - x_arr_train, x_arr_val, x_arr_test: These are the input features for the training, validation, and test sets, respectively, in NumPy array format. 
      They have the same shape as their corresponding PyTorch tensors.
      Shapes:
      - x_arr_train: {x_arr_train.shape}
      - x_arr_val: {x_arr_val.shape}
      - x_arr_test: {x_arr_test.shape}

    - y_arr_train, y_arr_val, y_arr_test: These are the target values for the training, validation, and test sets, respectively, in NumPy array format. 
      They have the same shape as their corresponding PyTorch tensors.
      Shapes:
      - y_arr_train: {y_arr_train.shape}
      - y_arr_val: {y_arr_val.shape}
      - y_arr_test: {y_arr_test.shape}
    """)

    return x_train, x_val, x_test, y_train, y_val, y_test, x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test

def calculate_features(feature_list, columns_to_drop, source_forecasted, feature_forecasted, sources):
    if columns_to_drop is None:
        columns_to_drop = []
    remaining_features = [feature for feature in feature_list if feature not in columns_to_drop]
    selected_feature = len(remaining_features) * source_forecasted + feature_forecasted
    num_features = len(sources) * len(remaining_features)
    return remaining_features, selected_feature, num_features

def preprocess_data(config_data, dataset_name):
    sources = config_data['sources']
    source_forecasted = config_data['source_forecasted']
    feature_list = config_data['feature_list']
    feature_forecasted = config_data['feature_forecasted']
    columns_to_drop = config_data.get('columns_to_drop') or []
    crop_start = config_data.get('crop_start')
    crop_end = config_data.get('crop_end')

    ts_past = config_data['ts_past']
    ts_future = config_data['ts_future']

    data_dir = get_path('data_dir', dataset_name)

    df = load_data(sources, data_dir, columns_to_drop)
    df = crop_data(df, crop_start, crop_end)
    df, z_mean, z_std = z_score_normalize(df)

    remaining_features, selected_feature, num_features = calculate_features(
        feature_list, columns_to_drop, source_forecasted, feature_forecasted, sources
    )

    x_arr, y_arr = split_sequences(df, df.iloc[:, selected_feature], ts_past, ts_future)

    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"x_arr shape: {x_arr.shape}")
    logging.info(f"y_arr shape: {y_arr.shape}")
    logging.info(f"z_mean: {z_mean}")
    logging.info(f"z_std: {z_std}")
    logging.info(f"selected_feature: {selected_feature}")
    logging.info(f"num_features: {num_features}")

    return df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features

