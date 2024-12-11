
import torch
from torch.utils.data import DataLoader, TensorDataset
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class DataLoaderCreator:
    """
    Handles the creation of PyTorch DataLoaders from preprocessed data.
    """

    def __init__(self):
        # TODO should this be kept if id not really have anything to initialize???

        # TODO -> maybe move / convert to a general utility class
        pass

    def create_data_loader(self, x, y, batch_size, shuffle=False):
        """
        Create a single data loader.

        Args:
            x (Tensor): Features.
            y (Tensor): Labels.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: A PyTorch DataLoader instance.
        """
        logger.info(f"Creating data loader with batch_size={batch_size}, shuffle={shuffle}")
        dataset = TensorDataset(x, y)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def create_data_loaders(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
        """
        Create data loaders for training, validation, and testing datasets.

        Args:
            x_train (Tensor): Training features.
            y_train (Tensor): Training labels.
            x_val (Tensor): Validation features.
            y_val (Tensor): Validation labels.
            x_test (Tensor): Testing features.
            y_test (Tensor): Testing labels.
            batch_size (int): Batch size for data loaders.

        Returns:
            dict: Dictionary containing train_loader, val_loader, and test_loader.
        """
        return {
            "train_loader": self.create_data_loader(x_train, y_train, batch_size, shuffle=True),
            "val_loader": self.create_data_loader(x_val, y_val, batch_size, shuffle=False),
            "test_loader": self.create_data_loader(x_test, y_test, batch_size, shuffle=False)
        }


# TODO -> should this be a singelton????
# Singleton instance
data_loader = DataLoaderCreator()
