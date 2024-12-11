import numpy as np
from utilities.logging_utils import get_logger

logger = get_logger(__name__)


class FixedDataSplitter:
    def __init__(self, data_split):
        """
        Initializes the DataSplitter with specified fractions for training, validation, and testing.

        Args:
            data_split (dict): Dictionary containing train, val, and optionally test fractions.
                               If test fraction is not provided, it is calculated as the remaining.
        """
        self.frac_train = data_split.get("frac_train")
        self.frac_val = data_split.get("frac_val")
        self.frac_test = data_split.get("frac_test", None)  # Optional; calculated as remaining if not provided
        # MAYBE add Offset so the start can be shifted?

        # Validate fractions
        if not self.frac_test:
            total = self.frac_train + self.frac_val
            if total > 1.0:
                raise ValueError("Training and validation fractions exceed 100%. Please adjust the configuration.")
            self.frac_test = 1.0 - total
        else:
            total = self.frac_train + self.frac_val + self.frac_test
            if total > 1.0:
                raise ValueError("Training, validation, and test fractions exceed 100%. Please adjust the configuration.")

        logger.info(f"FixedDataSplitter initialized with fractions: Train({self.frac_train}), "
                    f"Val({self.frac_val}), Test({self.frac_test})")

    def split(self, x, y=None):
        """
        Splits the data into training, validation, and test sets.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray, optional): Output labels. If None, splits only the input features.

        Returns:
            dict: Dictionary containing split arrays for train, validation, and test sets.
        """
        num_samples = x.shape[0]

        # Determine the effective dataset size based on provided fractions
        effective_samples = num_samples
        if self.frac_test is not None:
            effective_samples = int((self.frac_train + self.frac_val + self.frac_test) * num_samples)

        num_train = int(self.frac_train * effective_samples)
        num_val = int(self.frac_val * effective_samples)
        num_test = effective_samples - num_train - num_val

        # Split the features into train/validate/test
        x_arr_train = x[:num_train]
        x_arr_val = x[num_train:num_train + num_val]
        x_arr_test = x[num_train + num_val:num_train + num_val + num_test]

        logger.info(f"Features split: Train({num_train}), Val({num_val}), Test({num_test})")

        split_data = {
            "x_arr_train": x_arr_train,
            "x_arr_val": x_arr_val,
            "x_arr_test": x_arr_test
        }

        # Split the labels if provided
        if y is not None:
            y_arr_train = y[:num_train]
            y_arr_val = y[num_train:num_train + num_val]
            y_arr_test = y[num_train + num_val:num_train + num_val + num_test]

            logger.info("Labels split into train, validation, and test sets.")

            split_data.update({
                "y_arr_train": y_arr_train,
                "y_arr_val": y_arr_val,
                "y_arr_test": y_arr_test
            })

        return split_data

