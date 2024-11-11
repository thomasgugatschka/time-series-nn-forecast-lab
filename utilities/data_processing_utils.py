import torch
import logging
import numpy as np
import pandas as pd


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