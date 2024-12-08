import torch

from data_pipeline.base_data_pipeline import BaseDataPipeline
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class TransformerDataPipeline(BaseDataPipeline):
    def __init__(self, dataset_config, data_transformer, data_splitter):
        super().__init__(dataset_config=dataset_config, data_transformer=data_transformer, data_splitter=data_splitter)

        logger.info("Initializing data pipeline for Transformer-based model architectures.")

    """


    - Generate Sequences: Create variable-length or fixed-length sequences (if needed).
    - Add Positional Encoding: Integrate positional encodings to the sequences.


    """

    def preprocess(self, df):
        """
        Preprocessing pipeline for Transformer-based models.

        Args:
            df (pd.DataFrame): Input dataframe to preprocess.

        Returns:
            tuple: Preprocessed data (tensor_data, array_data, metadata)
        """

        metadata = {}

        logger.info("Starting the preprocessing pipeline for Transformer-based model architectures.")

        logger.info("Preprocessing Step: Cropping data")
        crop_start = self.dataset_config.get('crop_start')
        crop_end = self.dataset_config.get('crop_end')
        df = self.crop_data(df, crop_start, crop_end)

        logger.info("Preprocessing Step: Normalizing data")
        df, metadata_scaler = self.normalize(df, method="z_score")
        metadata.update(metadata_scaler)

        logger.info("Preprocessing Step: Calculating features")
        features = self.calculate_features()
        metadata.update(features)
        num_features = features['num_features']

        logger.info("Preprocessing Step: Splitting data into train, val, and test")
        split_data = self.data_splitter.split(df, None)  # Split features only; labels are derived later
        x_arr_train, x_arr_val, x_arr_test = (
            split_data['x_arr_train'], split_data['x_arr_val'], split_data['x_arr_test'])

        logger.info("Preprocessing Step: Adding positional encodings")
        x_arr_train = self.add_positional_encoding(x_arr_train, num_features)
        x_arr_val = self.add_positional_encoding(x_arr_val, num_features)
        x_arr_test = self.add_positional_encoding(x_arr_test, num_features)

        # In the Transformer case, labels (`y`) might be derived differently depending on the task
        logger.info("Preprocessing Step: Generating labels")
        y_arr_train, y_arr_val, y_arr_test = self.generate_labels(x_arr_train, x_arr_val, x_arr_test)

        logger.info("Preprocessing Step: Converting to tensors")
        tensor_data = self.convert_to_tensors(
            x_arr_train=x_arr_train, x_arr_val=x_arr_val, x_arr_test=x_arr_test,
            y_arr_train=y_arr_train, y_arr_val=y_arr_val, y_arr_test=y_arr_test
        )
        x_train, x_val, x_test = tensor_data['x_train'], tensor_data['x_val'], tensor_data['x_test']
        y_train, y_val, y_test = tensor_data['y_train'], tensor_data['y_val'], tensor_data['y_test']

        tensor_data = {
            'x_train': x_train,
            'x_val': x_val,
            'x_test': x_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }

        array_data = {
            'x_arr_train': x_arr_train,
            'x_arr_val': x_arr_val,
            'x_arr_test': x_arr_test,
            'y_arr_train': y_arr_train,
            'y_arr_val': y_arr_val,
            'y_arr_test': y_arr_test,
        }

        logger.info("Preprocessing pipeline completed for Transformer-based model architectures.")

        return tensor_data, array_data, metadata

    def add_positional_encoding(self, x_arr, num_features):
        """
        Add positional encoding to input data.

        Args:
            x_arr (pd.DataFrame): Input feature array.
            num_features (int): Number of features.

        Returns:
            np.ndarray: Input array with positional encoding added.
        """
        # Convert DataFrame to a Numpy array
        if isinstance(x_arr, pd.DataFrame):
            x_arr = x_arr.to_numpy()

        seq_len = x_arr.shape[0]
        pe = torch.zeros(seq_len, num_features)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Ensure num_features is even for sinusoidal encoding
        if num_features % 2 != 0:
            logger.warning("Number of features is odd. Padding to make it even.")
            num_features += 1  # Increment to the nearest even number
            pe = torch.zeros(seq_len, num_features)  # Update positional encoding tensor

        div_term = torch.exp(
            torch.arange(0, num_features, 2).float() * (-torch.log(torch.tensor(10000.0)) / num_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe[:, :x_arr.shape[1]]  # Ensure the size matches the original array
        x_arr_with_pe = torch.tensor(x_arr) + pe
        return x_arr_with_pe.numpy()

    def generate_labels(self, x_train, x_val, x_test):
        """
        Generate labels for the Transformer model.

        Args:
            x_train, x_val, x_test (np.ndarray): Input features for train, val, and test splits.

        Returns:
            tuple: Labels (y_train, y_val, y_test) corresponding to the input splits.
        """
        ts_future = self.dataset_config['ts_future']

        # For forecasting, labels are the next `ts_future` steps
        def extract_labels(x):
            return x[:, -ts_future:]  # Assuming data is already windowed for transformers

        y_train = extract_labels(x_train)
        y_val = extract_labels(x_val)
        y_test = extract_labels(x_test)

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