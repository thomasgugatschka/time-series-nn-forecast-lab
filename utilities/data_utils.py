import os
import pandas as pd
import numpy as np
import yaml
import torch

from torch.utils.data import DataLoader, TensorDataset

from manager.path_manager import path_manager

from utilities.logging_utils import get_logger

logger = get_logger(__name__)


# TODO -> change to CLASSES so it is somewhat separated???
#

# TODO REFACOTR to use FIEL MANAGER
# Load configuration function
def load_config(config_file):
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    logger.info(f"Configuration loaded successfully from {config_file}")
    return config


# Create data loaders function
def create_data_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    # Create data loaders
    # TODO -> check what the shuffle does -> nad if TRUE is bad for Time Series Prediction
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# Load data function
def load_data(sources, data_dir, columns_to_drop):
    df_list = []
    for source in sources:
        target_file = os.path.join(data_dir, f"{source}.csv")
        logger.info(f"Loading data from {target_file}")
        df = pd.read_csv(target_file, encoding='utf-8')
        logger.info(f"Data shape before dropping columns: {df.shape}")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Data shape after dropping columns: {df.shape}")
        df_list.append(df)
        logger.info(f"Appended data from {source}.csv")
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


def load_and_preprocess_data(config_data, dataset, data_split):
    # Use the default split ratios
    frac_train = data_split['frac_train']
    frac_val = data_split['frac_val']

    # Extract split ratios from dataset-specific config if available, otherwise use the default split
    #frac_train = config_data.get('train_val_test_split', {}).get('frac_train', data_split['frac_train'])
    #frac_val = config_data.get('train_val_test_split', {}).get('frac_val', data_split['frac_val'])

    # Preprocess data
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = preprocess_data(config_data, dataset)

    # Split and convert data
    x_train, x_val, x_test, y_train, y_val, y_test, *_ = split_and_convert_data(x_arr, y_arr, frac_train=frac_train,
                                                                                frac_val=frac_val)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "num_features": num_features,
        "z_mean": z_mean,
        "z_std": z_std,
        "selected_feature": selected_feature
    }


# Split and convert data function
def split_and_convert_data(x_arr, y_arr, frac_train, frac_val):
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

    # Convert to tensors
    x_train = torch.tensor(x_arr_train, dtype=torch.float32)
    x_val = torch.tensor(x_arr_val, dtype=torch.float32)
    x_test = torch.tensor(x_arr_test, dtype=torch.float32)
    y_train = torch.tensor(y_arr_train, dtype=torch.float32)
    y_val = torch.tensor(y_arr_val, dtype=torch.float32)
    y_test = torch.tensor(y_arr_test, dtype=torch.float32)

    logger.info(f"""
    Data split and conversion completed:
    - Shapes:
      - x_train: {x_train.shape}
      - x_val: {x_val.shape}
      - x_test: {x_test.shape}

    - Shapes:
      - y_train: {y_train.shape}
      - y_val: {y_val.shape}
      - y_test: {y_test.shape}

    - Shapes:
      - x_arr_train: {x_arr_train.shape}
      - x_arr_val: {x_arr_val.shape}
      - x_arr_test: {x_arr_test.shape}

    - Shapes:
      - y_arr_train: {y_arr_train.shape}
      - y_arr_val: {y_arr_val.shape}
      - y_arr_test: {y_arr_test.shape}
    """)

    # TODO - Refactor return so it works the same way as for the load_and_preprocess_data function
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

    data_dir = path_manager.get_path_data(dataset_name=dataset_name)

    df = load_data(sources, data_dir, columns_to_drop)
    df = crop_data(df, crop_start, crop_end)
    df, z_mean, z_std = z_score_normalize(df)

    remaining_features, selected_feature, num_features = calculate_features(
        feature_list, columns_to_drop, source_forecasted, feature_forecasted, sources
    )

    x_arr, y_arr = split_sequences(df, df.iloc[:, selected_feature], ts_past, ts_future)

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"x_arr shape: {x_arr.shape}")
    logger.info(f"y_arr shape: {y_arr.shape}")
    logger.info(f"z_mean: {z_mean}")
    logger.info(f"z_std: {z_std}")
    logger.info(f"selected_feature: {selected_feature}")
    logger.info(f"num_features: {num_features}")

    return df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features
