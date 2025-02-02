import numpy as np
from src.utilities.logging_utils import get_logger

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
        self.frac_test_tmp = data_split.get("frac_test", None)  # Optional; calculated as remaining if not provided
        self.unused_split = data_split.get("unused_split", "beginning")  # Optional; unused data split


        # Validate fractions
        if not self.frac_test_tmp:
            total = self.frac_train + self.frac_val
            if total > 1.0:
                raise ValueError("Training and validation fractions exceed 100%. Please adjust the configuration.")
            self.frac_test = 1.0 - total
        else:
            total = self.frac_train + self.frac_val + self.frac_test_tmp
            if total > 1.0:
                raise ValueError("Training, validation, and test fractions exceed 100%. Please adjust the configuration.")
            self.frac_test = self.frac_test_tmp

        if self.unused_split not in ["beginning", "end"]:
            raise ValueError("Invalid value for 'unused_split'. Choose between 'beginning' or 'end'.")


        logger.info(f"FixedDataSplitter initialized with fractions: Train({self.frac_train}), "
                    f"Val({self.frac_val}), Test({self.frac_test}) and unused_split: {self.unused_split}")


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

        # Ensure the fractions sum up to 1.0 or less
        total_frac = self.frac_train + self.frac_val + self.frac_test
        if total_frac > 1.0:
            raise ValueError("Total of frac_train, frac_val, and frac_test exceeds 1.0.")

        # Compute absolute split sizes based on full dataset
        num_train = int(self.frac_train * num_samples)
        num_val = int(self.frac_val * num_samples)
        num_test = int(self.frac_test * num_samples)

        logger.info(f"total_frac: {total_frac} | frac_train: {self.frac_train} | frac_val: {self.frac_val} | frac_test: {self.frac_test}")
        logger.info(f"Number of samples: {num_samples} | Train({num_train}), Val({num_val}), Test({num_test})")

        # Ensure the split sum is within the available sample count
        used_samples = num_train + num_val + num_test
        unused_samples = num_samples - used_samples

        if unused_samples > 0:
            logger.info(f"Trimmed {unused_samples} samples from the {self.unused_split} due to fractional mismatch.")
            if self.unused_split == "beginning":
                x = x[unused_samples:]  # Trim from the beginning
                if y is not None:
                    y = y[unused_samples:]
            elif self.unused_split == "end":
                x = x[:used_samples]  # Trim from the end
                if y is not None:
                    y = y[:used_samples]
            else:
                raise ValueError("Invalid value for 'unused_split'. Choose between 'beginning' or 'end'.")

        # Perform the split using the corrected numbers
        x_arr_train = x[:num_train]
        x_arr_val = x[num_train:num_train + num_val]
        x_arr_test = x[num_train + num_val:num_train + num_val + num_test]

        split_data = {
            "x_arr_train": x_arr_train,
            "x_arr_val": x_arr_val,
            "x_arr_test": x_arr_test
        }

        if y is not None:
            y_arr_train = y[:num_train]
            y_arr_val = y[num_train:num_train + num_val]
            y_arr_test = y[num_train + num_val:num_train + num_val + num_test]

            split_data.update({
                "y_arr_train": y_arr_train,
                "y_arr_val": y_arr_val,
                "y_arr_test": y_arr_test
            })

        logger.info(f"Features split: Train({num_train}), Val({num_val}), Test({num_test})")

        return split_data