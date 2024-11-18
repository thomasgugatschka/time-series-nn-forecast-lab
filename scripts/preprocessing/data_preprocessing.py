import logging
from configs.paths import get_path
from configs.logging_config import setup_logging
from utilities.data_utils import load_config, preprocess_data

setup_logging()

# Test Script for data preprocessing -> to check if functions are working correctly
def main(config_data_file, dataset_name):
    config_data = load_config(config_data_file)

    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = preprocess_data(config_data, dataset_name)
    print("Data loaded and preprocessed successfully.")
    print(f"DataFrame shape: {df.shape}")
    print(f"x_arr shape: {x_arr.shape}")
    print(f"y_arr shape: {y_arr.shape}")

if __name__ == "__main__":
    config_data_file = get_path('config_dir', 'config_data_sinusoidal.yaml')
    main(config_data_file, 'sinusoidal')