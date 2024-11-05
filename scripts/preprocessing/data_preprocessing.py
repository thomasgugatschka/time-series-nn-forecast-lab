import os
import logging
import pandas as pd
from utilities.utilities import split_sequences
from configs.paths import get_path
from configs.logging_config import setup_logging

# TODO - use both ETTh and ETTm sets to train the model like below !!!
# TODO --> so use both sets to train and calculate the features / shape / ...

# TODO -> check if it makes more sense to move to "utilities"

# TODO -> REWRITE WITH NEW DATA YAML IN MIND!!!!


def load_and_preprocess_data(config_data, config_train):
    stations = config_data['stations']
    station_forecasted = config_data['station_forecasted']
    feature_list = config_data['feature_list']
    feature_forecasted = config_data['feature_forecasted']
    ts_past = config_train['ts_past']
    ts_future = config_train['ts_future']

    data_dir = get_path('data_dir', 'pickles') # TODO - add to config the model -> so the path to the correct data can be build

    selected_feature = len(feature_list) * station_forecasted + feature_forecasted
    num_features = len(stations) * len(feature_list)

    # Import time-series from stored pickles
    df_list = []
    files = os.listdir(data_dir)
    for file in files:
        target_file = os.path.join(data_dir, file)
        df = pd.read_pickle(target_file)
        df = df.drop(columns='Date/Time (LST)') # TODO set this through YAML/Config file
        df_list.append(df)

    df = pd.concat(df_list, axis=1)
    df = df.iloc[40000:60000, :]  # crop df to lower calc time # TODO - move this to config file

    # Preprocessing data with z-score normalization
    z_mean = df.mean()
    z_std = df.std()
    df = (df - z_mean) / z_std

    # Feed context and prediction results to the model (adding another dimension)
    x_arr, y_arr = split_sequences(df, df.iloc[:, selected_feature], ts_past, ts_future)

    return df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features