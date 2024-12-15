from data_pipeline.preparation.data_loader_creator import DataLoaderCreator
from data_pipeline.preparation.data_splitter import FixedDataSplitter
from data_pipeline.utils.data_transformation import DataTransformation
from data_pipeline.ingestion.raw_data_loader import RawDataLoader

from data_pipeline.utils.data_pipeline_factory import DataPipelineFactory

from manager.file_manager import file_manager
from manager.path_manager import path_manager

from utilities.logging_utils import get_logger

logger = get_logger(__name__)

# TODO - move DataManager to data_pipeline folder
# TODO -> change to SINGELTON class -> so the data is loaded only once and the DataTransformation is only created once
# TODO - so the sklearn keep their metadata information !!!!

class DataManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager

        self.file_manager = file_manager
        self.path_manager = path_manager

        # Load dataset-specific and model-specific configurations
        self.model_type = config_manager['model_type']
        self.dataset = config_manager['dataset']

        self.data_params = config_manager['data_params']

        self.dataset_config = config_manager.get_dataset_config()


        # TODO -> add somewhere the fixed-window or full-sequence option

        # Internal state for storing intermediate results
        self._raw_df = None
        self._tensor_data = None
        self._array_data = None
        self._metadata = None

        # Initialize components
        self.raw_data_loader = RawDataLoader()
        self.data_transformation = DataTransformation()
        self.data_loader = DataLoaderCreator()
        self.data_splitter = FixedDataSplitter(self.data_params.get('data_split'))
        self.pipeline_preprocessing = (DataPipelineFactory.create_data_pipeline(
            model_type=self.model_type,
            dataset_config=self.dataset_config,
            data_transformer=self.data_transformation,
            data_splitter=self.data_splitter,
            window_size=self.data_params.get('time_series_window'))
        )

        logger.info(f"DataManager initialized for dataset: {self.dataset} and model type: {self.model_type}")

    def load_raw_data(self):
        """
        Load data from sources using dataset-specific configuration.

        Returns:
            pd.DataFrame: Loaded raw dataframe.
        """

        data_dir = self.path_manager.get_path_data(dataset_name=self.dataset)
        sources = self.dataset_config['sources']
        columns_to_drop = self.dataset_config.get('columns_to_drop', [])

        logger.info("Loading data using RawDataLoader.")
        self._raw_df = self.raw_data_loader.load_data(data_dir, sources, columns_to_drop)
        logger.info(f"Data loaded with shape: {self._raw_df.shape}")
        logger.info("Data stored in memory for further processing.")

        return self._raw_df

    def preprocess_data(self, raw_df=None):
        """
        Apply the preprocessing pipeline to the raw data.

        Args:
            raw_df (pd.DataFrame): The raw dataframe to preprocess.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        if raw_df is not None:
            logger.info("Using overridden raw data for pipeline.")
            df = raw_df
        elif self._raw_df is not None:
            logger.info("Using cached raw df for pipeline.")
            df = self._raw_df
        else:
            raise ValueError("No raw df available for pipeline. Load data first or provide it explicitly.")

        self._tensor_data, self._array_data, self._metadata = self.pipeline_preprocessing.preprocess(df)


        logger.info(f"x_train shape: {self._tensor_data['x_train'].shape}, y_train shape: {self._tensor_data['y_train'].shape}")
        logger.info(f"x_val shape: {self._tensor_data['x_val'].shape}, y_val shape: {self._tensor_data['y_val'].shape}")

        preprocessed_data = {
            'tensor_data': self._tensor_data,
            'array_data': self._array_data,
            'metadata': self._metadata
        }
        return preprocessed_data

    def get_data(self, key):
        """
        Retrieve a specific component of the DataManager's internal data.

        Args:
            key (str): The key of the component to retrieve. Options are 'raw_data', 'tensor_data', 'array_data', or 'metadata'.

        Returns:
            dict or pd.DataFrame: The requested component of the data.

        Raises:
            ValueError: If the requested key is invalid or the requested component is not available.
        """
        valid_keys = {
            'raw_data': self._raw_df,
            'tensor_data': self._tensor_data,
            'array_data': self._array_data,
            'metadata': self._metadata
        }

        if key not in valid_keys:
            raise ValueError(f"Invalid key '{key}'. Valid keys are: {list(valid_keys.keys())}.")

        component = valid_keys[key]
        if component is None:
            raise ValueError(
                f"The requested component '{key}' is not available. Ensure the corresponding data is loaded or processed first.")

        return component

    def create_data_loaders(self, tensor_data=None, batch_size=None, loaders_to_create=None, shuffle_train=False):
        """
        Create data loaders for the model training.

        Args:
            tensor_data (dict, optional): Preprocessed data containing x_train, y_train, x_val, y_val, x_test, y_test.
            batch_size (int): Batch size for the data loaders. This must always be provided.
            loaders_to_create (list, optional): List specifying which loaders to create.
                                                Options: ["train", "val", "test"]. Defaults to all.
            shuffle_train (bool, optional): Whether to shuffle the training data loader.

        Returns:
            dict: Dictionary containing the requested data loaders.
        """
        if batch_size is None:
            raise ValueError("Batch size must be provided to create data loaders.")

        if tensor_data is None and self._tensor_data is None:
            raise ValueError(
                "Tensor data is not available. Provide tensor_data or run preprocessing first."
            )

        # Use provided preprocessed_data or fallback to stored preprocessed data
        tensor_data = tensor_data or self._tensor_data
        loaders_to_create = loaders_to_create or ["train", "val", "test"]

        split_loaders = {}

        try:
            if "train" in loaders_to_create:
                split_loaders["train_loader"] = self.data_loader.create_data_loader(
                    tensor_data["x_train"], tensor_data["y_train"], batch_size, shuffle=shuffle_train
                )

            if "val" in loaders_to_create:
                split_loaders["val_loader"] = self.data_loader.create_data_loader(
                    tensor_data["x_val"], tensor_data["y_val"], batch_size, shuffle=False
                )

            if "test" in loaders_to_create:
                split_loaders["test_loader"] = self.data_loader.create_data_loader(
                    tensor_data["x_test"], tensor_data["y_test"], batch_size, shuffle=False
                )
        except KeyError as e:
            raise KeyError(f"Missing required key in preprocessed data: {e}")

        if not split_loaders:
            raise ValueError(f"No loaders were created. Check the 'loaders_to_create' parameter: {loaders_to_create}")

        logger.info(f"Successfully created data loaders for: {list(split_loaders.keys())}")
        return split_loaders

    def clear_memory(self):
        """
        Clear all in-memory data variables.
        """
        logger.info("Clearing all in-memory data variables.")
        self._raw_df = None
        self._tensor_data = None
        self._array_data = None
        self._metadata = None


