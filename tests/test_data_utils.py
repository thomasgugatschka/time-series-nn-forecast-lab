# tests/test_data_preprocessing.py
import unittest
import numpy as np
import pandas as pd
from utilities.data_utils import split_sequences, z_score_normalize, crop_data

class TestDataPreprocessing(unittest.TestCase):

    def test_crop_data(self):
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20)
        })

        # Test cropping with both start and end
        cropped_df = crop_data(df, 2, 5)
        expected_df = pd.DataFrame({
            'A': [2, 3, 4],
            'B': [12, 13, 14]
        }, index=[2, 3, 4])
        pd.testing.assert_frame_equal(cropped_df, expected_df)

        # Test cropping with only start
        cropped_df = crop_data(df, 7, None)
        expected_df = pd.DataFrame({
            'A': [7, 8, 9],
            'B': [17, 18, 19]
        }, index=[7, 8, 9])
        pd.testing.assert_frame_equal(cropped_df, expected_df)

        # Test cropping with only end
        cropped_df = crop_data(df, None, 3)
        expected_df = pd.DataFrame({
            'A': [0, 1, 2],
            'B': [10, 11, 12]
        }, index=[0, 1, 2])
        pd.testing.assert_frame_equal(cropped_df, expected_df)

        # Test cropping with neither start nor end
        cropped_df = crop_data(df, None, None)
        pd.testing.assert_frame_equal(cropped_df, df)

    def test_split_sequences(self):
        input_sequences = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        output_sequence = np.array([1, 2, 3, 4, 5])
        n_steps_in, n_steps_out = 2, 2
        x, y = split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out)
        expected_x = np.array([[[1, 2], [3, 4]], [[3, 4], [5, 6]], [[5, 6], [7, 8]]])
        expected_y = np.array([[2, 3], [3, 4], [4, 5]])
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)

    def test_z_score_normalize(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        normalized_df, z_mean, z_std = z_score_normalize(df)
        expected_normalized_df = (df - df.mean()) / df.std()
        pd.testing.assert_frame_equal(normalized_df, expected_normalized_df)
        pd.testing.assert_series_equal(z_mean, df.mean())
        pd.testing.assert_series_equal(z_std, df.std())

if __name__ == '__main__':
    unittest.main()