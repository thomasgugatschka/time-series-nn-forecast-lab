import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

    # Z-Score Methods
    def apply_z_score(self, df):
        """
        Applies z-score normalization to the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to normalize.

        Returns:
            tuple: Normalized dataframe and metadata for reversing the transformation.
        """
        self.validate_dataframe(df)
        z_mean = df.mean()
        z_std = df.std()
        metadata = {"z_mean": z_mean, "z_std": z_std}
        normalized_df = (df - z_mean) / z_std
        return normalized_df, metadata

    def reverse_z_score(self, df, metadata):
        """
        Reverses z-score normalization.

        Args:
            df (pd.DataFrame): The normalized dataframe to denormalize.
            metadata (dict): Metadata containing mean and std deviation for each column.

        Returns:
            pd.DataFrame: The denormalized dataframe.
        """
        z_mean = metadata["z_mean"]
        z_std = metadata["z_std"]
        return (df * z_std) + z_mean

    # Min-Max Methods
    def apply_min_max(self, df, feature_range=(0, 1)):
        """
        Applies min-max scaling to the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to scale.
            feature_range (tuple): Desired range of transformed data. Defaults to (0, 1).

        Returns:
            tuple: Scaled dataframe and metadata for reversing the transformation.
        """
        self.validate_dataframe(df)
        min_vals = df.min()
        max_vals = df.max()
        min_range, max_range = feature_range

        scaled_df = ((df - min_vals) / (max_vals - min_vals)) * (max_range - min_range) + min_range
        metadata = {"min_vals": min_vals, "max_vals": max_vals, "min_range": min_range, "max_range": max_range}
        return scaled_df, metadata

    def reverse_min_max(self, df, metadata):
        """
        Reverses min-max scaling.

        Args:
            df (pd.DataFrame): The scaled dataframe to revert.
            metadata (dict): Metadata containing min and max values for each feature, as well as feature range.

        Returns:
            pd.DataFrame: The dataframe scaled back to its original values.
        """
        min_vals = metadata["min_vals"]
        max_vals = metadata["max_vals"]
        min_range = metadata["min_range"]
        max_range = metadata["max_range"]

        original_df = ((df - min_range) / (max_range - min_range)) * (max_vals - min_vals) + min_vals
        return original_df

    # Optional sklearn-based methods
    def apply_scaler(self, df, method):
        """
        Applies a scaler from sklearn for normalization or scaling.

        Args:
            df (pd.DataFrame): The dataframe to scale.
            method (str): The scaler method, either 'z_score' or 'min_max'.

        Returns:
            tuple: Scaled dataframe and the fitted scaler for reversing the transformation.
        """
        self.validate_dataframe(df)
        if method not in self.scalers:
            raise ValueError(f"Unsupported scaling method: {method}")

        scaler = self.scalers[method]
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return scaled_df, scaler

    def reverse_scaler(self, df, scaler):
        """
        Reverses a transformation applied using sklearn scaler.

        Args:
            df (pd.DataFrame): The scaled dataframe.
            scaler (sklearn Scaler): The fitted scaler used for transformation.

        Returns:
            pd.DataFrame: The dataframe scaled back to its original values.
        """
        if not scaler:
            raise ValueError("Scaler instance is required to reverse transformation.")
        return pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
