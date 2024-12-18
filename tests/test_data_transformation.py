import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_pipeline.utils.data_transformation import DataTransformation


class TestDataTransformation(unittest.TestCase):
    def setUp(self):
        """Set up the test environment with sample data and a DataTransformation instance."""
        self.data_transformation = DataTransformation()

        # Example DataFrame with multiple features
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })

    def test_apply_scaler_standard_scaler(self):
        """Test apply_scaler with StandardScaler on a full dataset."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="z_score")

        # Assertions
        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(scaled_df.shape, self.df.shape)
        np.testing.assert_almost_equal(scaled_df.mean().values, [0, 0], decimal=5)
        np.testing.assert_almost_equal(scaled_df.std().values, [1, 1], decimal=5)

    def test_reverse_scaler_standard_scaler_full(self):
        """Test reverse_scaler with StandardScaler on a full dataset."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="z_score")
        reversed_df = self.data_transformation.reverse_scaler(scaled_df, scaler, selected=False)

        # Assertions
        self.assertIsInstance(reversed_df, pd.DataFrame)
        pd.testing.assert_frame_equal(reversed_df, self.df)

    def test_reverse_scaler_standard_scaler_single_feature(self):
        """Test reverse_scaler with StandardScaler on a single feature."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="z_score")

        # Select only the first feature
        single_feature = scaled_df["feature1"].values.reshape(-1, 1)

        # Reverse scaling for the selected feature
        reversed_feature = self.data_transformation.reverse_scaler(
            single_feature, scaler, selected=True, feature=0
        )

        # Assertions
        np.testing.assert_array_almost_equal(reversed_feature.flatten(), self.df["feature1"].values)

    def test_apply_scaler_minmax_scaler(self):
        """Test apply_scaler with MinMaxScaler on a full dataset."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="min_max")

        # Assertions
        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(scaled_df.shape, self.df.shape)
        np.testing.assert_almost_equal(scaled_df.min().values, [0, 0], decimal=5)
        np.testing.assert_almost_equal(scaled_df.max().values, [1, 1], decimal=5)

    def test_reverse_scaler_minmax_scaler_full(self):
        """Test reverse_scaler with MinMaxScaler on a full dataset."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="min_max")
        reversed_df = self.data_transformation.reverse_scaler(scaled_df, scaler, selected=False)

        # Assertions
        self.assertIsInstance(reversed_df, pd.DataFrame)
        pd.testing.assert_frame_equal(reversed_df, self.df)

    def test_reverse_scaler_minmax_scaler_single_feature(self):
        """Test reverse_scaler with MinMaxScaler on a single feature."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="min_max")

        # Select only the second feature
        single_feature = scaled_df["feature2"].values.reshape(-1, 1)

        # Reverse scaling for the selected feature
        reversed_feature = self.data_transformation.reverse_scaler(
            single_feature, scaler, selected=True, feature=1
        )

        # Assertions
        np.testing.assert_array_almost_equal(reversed_feature.flatten(), self.df["feature2"].values)

    def test_invalid_scaler_method(self):
        """Test apply_scaler with an invalid scaler method."""
        with self.assertRaises(ValueError):
            self.data_transformation.apply_scaler(self.df, method="invalid_scaler")

    def test_reverse_scaler_no_scaler(self):
        """Test reverse_scaler without providing a scaler."""
        with self.assertRaises(ValueError):
            self.data_transformation.reverse_scaler(self.df, scaler=None, selected=False)

    def test_reverse_scaler_no_selected_feature(self):
        """Test reverse_scaler when selected=True but no feature is provided."""
        scaled_df, scaler = self.data_transformation.apply_scaler(self.df, method="z_score")
        single_feature = scaled_df["feature1"].values.reshape(-1, 1)

        with self.assertRaises(ValueError):
            self.data_transformation.reverse_scaler(single_feature, scaler, selected=True, feature=None)



if __name__ == '__main__':
    unittest.main()
