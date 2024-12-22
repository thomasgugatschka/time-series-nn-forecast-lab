import numpy as np
import pandas as pd
import torch

from data_pipeline.preprocessors.base_data_pipeline import BaseDataPipeline
from utilities.logging_utils import get_logger

logger = get_logger(__name__)


# TODO -> move out to it's own file
class TimeSeriesWindowGenerator:
    @staticmethod
    def generate_fixed_time_series_windows(input_data, target_data, ts_past, ts_future=None):
        """
        Generate fixed time-series windows for features (input) and labels (target).

        Args:
            input_data (np.ndarray): Input data array with shape (num_samples, num_features).
            target_data (np.ndarray): Target data array with shape (num_samples,) for a single feature.
            ts_past (int): Number of time steps in the input sequence.
            ts_future (int, optional): Number of time steps to predict. If None, no target windows are created.

        Returns:
            tuple: Arrays for input sequences (`x`) and output sequences (`y` if `ts_future` is provided).
        """
        logger.info(f"Generating time-series windows with ts_past={ts_past}, ts_future={ts_future or 0}")
        x, y = [], []

        for i in range(len(input_data) - ts_past - (ts_future or 0) + 1):
            x_window = input_data[i:i + ts_past]  # Input sequence
            x.append(x_window)

            if ts_future:
                y_window = target_data[i + ts_past:i + ts_past + ts_future]  # Target sequence
                y.append(y_window)

        logger.info(f"Generated {len(x)} windows for input.")
        if ts_future:
            logger.info(f"Generated {len(y)} windows for target.")
            return np.array(x), np.array(y)
        return np.array(x), None


class TransformerDataPipeline(BaseDataPipeline):
    def __init__(self, dataset_config, data_transformer, data_splitter, window_size=None, **kwargs):
        super().__init__(dataset_config=dataset_config, data_transformer=data_transformer, data_splitter=data_splitter)

        if window_size is None:
            self.mode = 'fixed_window'
        self.mode = window_size

        logger.info(f"Initializing Transformer pipeline in {self.mode} mode.")

    def preprocess(self, df: pd.DataFrame):
        logger.info("Starting preprocessing pipeline for Transformer-based models.")
        metadata = {}


        # Step 1: Crop and normalize the data
        crop_start = self.dataset_config.get('crop_start')
        crop_end = self.dataset_config.get('crop_end')
        cropped_df = self.crop_data(df, crop_start, crop_end)
        df, scaler = self.normalize(cropped_df, method="z_score")

        metadata.update({'crop_start': crop_start, 'crop_end': crop_end})
        # Step 2: Calculate features
        features = self.calculate_features()
        metadata.update(features)
        selected_feature, num_features = features['selected_feature'], features['num_features']

        # Step 3: Handle splitting and label generation
        ts_past = self.dataset_config['ts_past']
        ts_future = self.dataset_config['ts_future']

        if self.mode == 'fixed-window':
            logger.info("Generating fixed time series windows.")
            x_arr, y_arr = TimeSeriesWindowGenerator.generate_fixed_time_series_windows(
                input_data=df.to_numpy(),
                target_data=df.iloc[:, selected_feature].to_numpy(),  # Use only the selected feature as the target
                ts_past=ts_past,
                ts_future=ts_future
            )
            logger.info(f"x_arr shape: {x_arr.shape}, y_arr shape: {y_arr.shape}")
            split_data = self.data_splitter.split(x_arr, y_arr)
            # Step 3: Handle splitting and label generation
        elif self.mode == 'full-sequence':
            logger.info("Using full-sequence mode.")

            split_data = self.data_splitter.split(df.to_numpy(), None)

            x_arr_train, x_arr_val, x_arr_test = (
                split_data['x_arr_train'], split_data['x_arr_val'], split_data['x_arr_test']
            )

            # Generate labels for the selected feature
            logger.info(
                f"x_arr_train shape: {x_arr_train.shape}, x_arr_val shape: {x_arr_val.shape}, x_arr_test shape: {x_arr_test.shape}")
            x_arr_train, y_arr_train, x_arr_val, y_arr_val, x_arr_test, y_arr_test = self.generate_labels_full_sequence(
                x_arr_train, x_arr_val, x_arr_test, selected_feature
            )

            # Truncate x to match y
 #           x_arr_train = x_arr_train[:len(y_arr_train)]
 #           x_arr_val = x_arr_val[:len(y_arr_val)]
 #           x_arr_test = x_arr_test[:len(y_arr_test)]

            split_data.update({
                "x_arr_train": x_arr_train,
                "x_arr_val": x_arr_val,
                "x_arr_test": x_arr_test,
                "y_arr_train": y_arr_train,
                "y_arr_val": y_arr_val,
                "y_arr_test": y_arr_test,
            })
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Log shapes for debugging
        logger.info(f"x_train shape: {split_data['x_arr_train'].shape}, y_train shape: {split_data['y_arr_train'].shape}")
        logger.info(f"x_val shape: {split_data['x_arr_val'].shape}, y_val shape: {split_data['y_arr_val'].shape}")
        logger.info(f"x_test shape: {split_data['x_arr_test'].shape}, y_test shape: {split_data['y_arr_test'].shape}")

        # Step 4: Add positional encodings
        logger.info("Adding positional encodings.")
        split_data['x_arr_train'] = self.add_positional_encoding(split_data['x_arr_train'], num_features)
        split_data['x_arr_val'] = self.add_positional_encoding(split_data['x_arr_val'], num_features)
        split_data['x_arr_test'] = self.add_positional_encoding(split_data['x_arr_test'], num_features)


        # Step 5: Convert to tensors
        logger.info("Converting to tensors.")
        tensor_data = self.convert_to_tensors(
            x_arr_train=split_data['x_arr_train'],
            x_arr_val=split_data['x_arr_val'],
            x_arr_test=split_data['x_arr_test'],
            y_arr_train=split_data['y_arr_train'],
            y_arr_val=split_data['y_arr_val'],
            y_arr_test=split_data['y_arr_test']
        )

        logger.info("Preprocessing pipeline completed for Transformer-based models.")
        return cropped_df, tensor_data, split_data, scaler, metadata

    def add_positional_encoding(self, x_arr, num_features):
        if isinstance(x_arr, pd.DataFrame):
            x_arr = x_arr.to_numpy()

        if x_arr.ndim == 2:
            x_arr = x_arr[:, None, :]  # Add seq_len=1 dimension if missing

        seq_len = x_arr.shape[1]

        logger.info(f"Adding positional encodings to input data with shape: {x_arr.shape}")
        logger.info(f"Sequence length: {seq_len}, Number of features: {num_features}")

        pe = torch.zeros(seq_len, num_features)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        logger.info(f"Position shape: {position.shape}, PE shape: {pe.shape}")

        if num_features % 2 != 0:
            num_features += 1
            pe = torch.zeros(seq_len, num_features)

        div_term = torch.exp(torch.arange(0, num_features, 2).float() * (-torch.log(torch.tensor(10000.0)) / num_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:seq_len, :x_arr.shape[2]]

        return (torch.tensor(x_arr) + pe).numpy()

    def generate_labels_full_sequence(self, x_train, x_val, x_test, selected_feature):
        ts_past = self.dataset_config['ts_past']
        ts_future = self.dataset_config['ts_future']

        def reshape_and_extract(x):
            # Ensure input data has enough samples
            num_samples = len(x) - ts_past - ts_future + 1
            if num_samples <= 0:
                raise ValueError(f"Insufficient data points for ts_past={ts_past} and ts_future={ts_future}.")

            logger.info(f"Reshaping input with shape {x.shape} into windows for labels.")

            # Create input windows (x_reshaped) of shape (num_samples, ts_past, num_features)
            x_reshaped = np.array([x[i:i + ts_past] for i in range(num_samples)])

            # Create target sequences (y_reshaped) of shape (num_samples, ts_future)
            y_reshaped = np.array(
                [x[i + ts_past:i + ts_past + ts_future, selected_feature] for i in range(num_samples)])

            logger.info(f"Generated labels with shapes: x_reshaped={x_reshaped.shape}, y_reshaped={y_reshaped.shape}")

            return x_reshaped, y_reshaped

        # Apply windowing to training, validation, and test splits
        x_train, y_train = reshape_and_extract(x_train)
        x_val, y_val = reshape_and_extract(x_val)
        x_test, y_test = reshape_and_extract(x_test)

        logger.info(
            f"Generated labels with shapes: x_train={x_train.shape}, y_train={y_train.shape}, "
            f"x_val={x_val.shape}, y_val={y_val.shape}, x_test={x_test.shape}, y_test={y_test.shape}"
        )

        # Return both x_reshaped and y_reshaped
        return x_train, y_train, x_val, y_val, x_test, y_test




    def generate_labels(self, x_train, x_val, x_test):
        ts_future = self.dataset_config['ts_future']

        def extract_labels(x):
            if x.shape[1] < ts_future:
                logger.error(
                    f"Cannot extract labels: Sequence length ({x.shape[1]}) is smaller than ts_future ({ts_future}).")
                return None
            logger.info(f"Extracting labels for input with shape: {x.shape}")
            return x[:, -ts_future:]

        y_train = extract_labels(x_train)
        y_val = extract_labels(x_val)
        y_test = extract_labels(x_test)

        if y_train is None or y_val is None or y_test is None:
            logger.error("Label generation failed for one or more splits.")

        return y_train, y_val, y_test

    def postprocess(self, predictions, metadata):
        """
        Postprocessing pipeline for Transformer-based models.

        Args:
            predictions (pd.DataFrame): Model predictions to postprocess.
            metadata (dict): Metadata for postprocessing.

        Returns:
            pd.DataFrame: Postprocessed predictions.
        """
        # Denormalize predictions
        logger.info("Postprocessing Step: Denormalizing predictions")
        df = self.denormalize(predictions, metadata, method="z_score")
        return df










