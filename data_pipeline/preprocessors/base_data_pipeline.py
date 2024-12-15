from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np

from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class BaseDataPipeline(ABC):
    """
    Abstract base class for all data pipelines. Provides common preprocessing and
    postprocessing methods while enforcing the implementation of architecture-specific methods.
    """

    def __init__(self, dataset_config, data_transformer, data_splitter):
        self.dataset_config = dataset_config
        self.data_transformer = data_transformer
        self.data_splitter = data_splitter

    @abstractmethod
    def preprocess(self, df):
        """
        Abstract method for preprocessing, to be implemented by subclasses.

        Args:
            df (pd.DataFrame): Input dataframe to preprocess.
        """
        pass

    @abstractmethod
    def postprocess(self, predictions, metadata):
        """
        Abstract method for postprocessing, to be implemented by subclasses.

        Args:
            predictions (pd.DataFrame): Model predictions to postprocess.
            metadata (dict): Metadata for postprocessing.
        """
        pass

    @staticmethod
    def crop_data(df, start=None, end=None):
        """
        Crops the dataframe between start and end indices.

        Args:
            df (pd.DataFrame): The dataframe to crop.
            start (int, optional): Starting index.
            end (int, optional): Ending index.

        Returns:
            pd.DataFrame: The cropped dataframe.
        """
        if start is not None and end is not None:
            logger.info(f"Cropping data from index {start} to {end}")
            df = df.iloc[start:end, :]
        elif start is not None:
            logger.info(f"Cropping data from index {start} onwards")
            df = df.iloc[start:, :]
        elif end is not None:
            logger.info(f"Cropping data up to index {end}")
            df = df.iloc[:end, :]
        return df

    @staticmethod
    def convert_to_tensors(x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test):
        """
        Converts the input numpy arrays to PyTorch tensors and returns them in a dictionary.

        Args:
            x_arr_train (np.ndarray): Training features.
            x_arr_val (np.ndarray): Validation features.
            x_arr_test (np.ndarray): Testing features.
            y_arr_train (np.ndarray): Training labels.
            y_arr_val (np.ndarray): Validation labels.
            y_arr_test (np.ndarray): Testing labels.

        Returns:
            dict: A dictionary of PyTorch tensors with keys corresponding to the input data names.
        """

        if any(v is None for v in [x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test]):
            raise ValueError("One or more input arrays are None. Check the preprocessing steps.")

        return {
            "x_train": torch.tensor(x_arr_train, dtype=torch.float32),
            "x_val": torch.tensor(x_arr_val, dtype=torch.float32),
            "x_test": torch.tensor(x_arr_test, dtype=torch.float32),
            "y_train": torch.tensor(y_arr_train, dtype=torch.float32),
            "y_val": torch.tensor(y_arr_val, dtype=torch.float32),
            "y_test": torch.tensor(y_arr_test, dtype=torch.float32),
        }

    @staticmethod
    def generate_fixed_time_series_windows(input_sequences, output_sequence, ts_past, ts_future):
        """
        Split sequences from input DataFrame into features and labels for model training.

        #TODO -> add docstring with new inputs
        # TODO -> explain the output sequence!!!!

        Args:
            df (pd.DataFrame): The dataframe to split.
            ts_past (int): Number of input time steps.
            ts_future (int): Number of output time steps.

        Returns:
            tuple: Arrays of input sequences (x) and output sequences (y).
        """
        logger.info(f"Splitting sequences with n_steps_in: {ts_past}, n_steps_out: {ts_future}")

        x, y = list(), list()

        # Iterate over the full length of input sequences to create time windows
        for i in range(len(input_sequences)):
            end_ix = i + ts_past                  # Determine the end index for the input sequence
            out_end_ix = end_ix + ts_future - 1    # Determine the end index for the output sequence
            if out_end_ix > len(input_sequences):       # If the output end index exceeds the number of available samples, stop iteration
                break

            seq_x = input_sequences[i:end_ix]               # Extract the input sequence from current index to the calculated end index
            seq_y = output_sequence[end_ix - 1:out_end_ix]  # Extract the output sequence starting from the last input step to the calculated output end index

            x.append(seq_x)
            y.append(seq_y)

        logger.info(f"Sequence splitting complete. Generated {len(x)} training pairs.")

        # Convert the lists of sequences to DataFrames and return them as a tuple (x, y)
        # TODO np oder eher mit pd.DataFrame? !!!!!
        return np.array(x), np.array(y)

    def calculate_features(self):
        """
        TODO: Add docstring
        """
        feature_list = self.dataset_config['feature_list']
        columns_to_drop = self.dataset_config['columns_to_drop']
        source_forecasted = self.dataset_config['source_forecasted']
        feature_forecasted = self.dataset_config['feature_forecasted']
        sources = self.dataset_config['sources']

        if columns_to_drop is None:
            columns_to_drop = []
        remaining_features = [feature for feature in feature_list if feature not in columns_to_drop]
        selected_feature = len(remaining_features) * source_forecasted + feature_forecasted
        num_features = len(sources) * len(remaining_features)

        return {'selected_feature': selected_feature, 'num_features': num_features}

    def split_data(self, x_arr, y_arr):
        """
        Split data into training, validation, and test sets.

        Args:
            x_arr (np.ndarray): Feature data.
            y_arr (np.ndarray): Target data.

        Returns:
            tuple: Split datasets as tensors (x_train, x_val, x_test, y_train, y_val, y_test).
        """

        logger.info("Splitting data into train, validation, and test sets.")
        return self.data_splitter.split(x_arr, y_arr)

    def normalize(self, df, method):
        # TODO: change over to SCALER -> first confirm if no dependencies to metadata for z-scor / min-max
        """Apply normalization."""
        if method == "z_score":
            logger.info("Applying z-score normalization")
            return self.data_transformer.apply_z_score(df)
        elif method == "min_max":
            logger.info("Applying min-max normalization")
            return self.data_transformer.apply_min_max(df)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def denormalize(self, df, metadata, method):
        """Apply denormalization."""
        if method == "z_score":
            logger.info("Reversing z-score normalization")
            return self.data_transformer.reverse_z_score(df, metadata)
        elif method == "min_max":
            logger.info("Reversing min-max normalization")
            return self.data_transformer.reverse_min_max(df, metadata)
        else:
            raise ValueError(f"Unsupported denormalization method: {method}")

