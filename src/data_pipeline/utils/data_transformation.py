import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)


class DataTransformation:
    """
    Handles data transformations, both to transform to a processed state
    and back-transform to the original state, with support for z-score and min-max scaling.
    """

    def __init__(self):
        # Initialize scalers as needed for optional use
        self.scalers = {
            "z_score": StandardScaler(),
            "min_max": MinMaxScaler()
        }

    @staticmethod
    def validate_dataframe(df):
        """
        Validate the input dataframe for common issues.

        Args:
            df (pd.DataFrame): The dataframe to validate.

        Raises:
            ValueError: If the dataframe is empty or contains invalid values.
        """
        if df.empty:
            raise ValueError("The input dataframe is empty.")
        if df.isnull().values.any():
            raise ValueError("The input dataframe contains NaN values.")

    def apply_scaler(self, df, method):
        """
        Applies a scaler from sklearn for normalization or scaling.
        """
        self.validate_dataframe(df)

        if method not in self.scalers:
            raise ValueError(f"Unsupported scaling method: {method}. Supported methods: {list(self.scalers.keys())}")

        scaler = self.scalers[method]
        scaled_array = scaler.fit_transform(df)

        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(scaled_array, columns=df.columns), scaler
        else:
            return scaled_array, scaler

    def reverse_scaler(self, data, scaler, selected=True, feature=None):
        """
        Reverses a transformation (denormalizes) using the provided scaler.

        Args:
            data (np.ndarray or pd.DataFrame): Scaled data to be denormalized.
            scaler: Fitted sklearn scaler (StandardScaler or MinMaxScaler).
            selected (bool): If True, denormalize only a selected feature.
            feature (int, optional): Index of the selected feature to denormalize.

        Returns:
            np.ndarray or pd.DataFrame: Denormalized data, preserving DataFrame structure if input was a DataFrame.

        Raises:
            ValueError: If the scaler is not provided or required arguments are missing.
        """
        if scaler is None:
            raise ValueError("Scaler is not provided. Ensure preprocessing has been done.")

        # Check if input is DataFrame and extract column names
        is_dataframe = isinstance(data, pd.DataFrame)
        columns = data.columns if is_dataframe else None

        # Convert data to NumPy array if it's a DataFrame
        data_array = data.to_numpy() if is_dataframe else data

        if selected:
            # Validate selected_feature
            if feature is None:
                raise ValueError("selected_feature must be specified when selected=True.")

            # Get the column name for the selected feature (if DataFrame)
            column_name = columns[feature] if is_dataframe else f"Feature_{feature}"
            logger.info(f"Denormalizing selected feature at index {feature} (Column: {column_name}).")

            # Perform manual denormalization for a single feature
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):  # StandardScaler
                mean = scaler.mean_[feature]
                scale = scaler.scale_[feature]
                denorm_data = (data_array * scale) + mean

            elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):  # MinMaxScaler
                min_val = scaler.min_[feature]
                scale = scaler.scale_[feature]
                feature_range_min = 0  # Default R_min in MinMaxScaler

                # Reverse MinMax scaling formula
                denorm_data = (data_array - feature_range_min) / scale + min_val
            else:
                raise ValueError("Unsupported scaler type for single feature denormalization.")

        else:
            # Denormalize the full dataset
            logger.info("Denormalizing the full dataset using scaler's inverse_transform.")
            denorm_data = scaler.inverse_transform(data_array)

        # Convert back to DataFrame if input was a DataFrame
        if is_dataframe:
            denorm_data = pd.DataFrame(denorm_data, columns=columns)

        return denorm_data

