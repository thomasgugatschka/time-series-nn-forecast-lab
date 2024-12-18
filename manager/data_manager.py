from data_pipeline.preparation.data_loader_creator import DataLoaderCreator
from data_pipeline.preparation.data_splitter import FixedDataSplitter
from data_pipeline.utils.data_transformation import DataTransformation
from data_pipeline.ingestion.raw_data_loader import RawDataLoader

from data_pipeline.utils.data_pipeline_factory import DataPipelineFactory

from manager.file_manager import file_manager
from manager.path_manager import path_manager

from utilities.logging_utils import get_logger

logger = get_logger(__name__)


class DataManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_manager):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.config_manager = config_manager
        self.file_manager = file_manager
        self.path_manager = path_manager

        self.dataset_config = config_manager.get_dataset_config()
        self.main_config = config_manager.get_main_config()

        self.model_type = self.main_config['model_type']
        self.dataset = self.main_config['dataset']
        self.data_params = self.main_config['data_params']

        # Internal state for storing intermediate results
        self._raw_df = None
        self._cropped_df = None
        self._tensor_data = None
        self._array_data = None
        self._scaler = None
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

        self._initialized = True
        logger.info(f"DataManager initialized for dataset: {self.dataset} and model type: {self.model_type}")

    def load_raw_data(self):
        """
        Load data from sources using dataset-specific configuration.

        Returns:
            pd.DataFrame: Loaded raw dataframe.
        """
        if self._raw_df is None:
            data_dir = self.path_manager.get_path_data(dataset_name=self.dataset)
            sources = self.dataset_config['sources']
            columns_to_drop = self.dataset_config.get('columns_to_drop', [])

            logger.info("Loading data using RawDataLoader.")
            self._raw_df = self.raw_data_loader.load_data(data_dir, sources, columns_to_drop)
            logger.info(f"Data loaded with shape: {self._raw_df.shape}")
        else:
            logger.info("Using cached raw data.")
        return self._raw_df

    def preprocess_data(self, raw_df=None):
        """
        Apply the preprocessing pipeline to the raw data.

        Args:
            raw_df (pd.DataFrame): The raw dataframe to preprocess.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        if self._tensor_data is None or raw_df is not None:
            logger.info("Starting data preprocessing pipeline.")
            df = raw_df if raw_df is not None else self._raw_df
            if df is None:
                raise ValueError("No raw data available. Load data first or provide a dataframe.")

            self._cropped_df, self._tensor_data, self._array_data, self._scaler, self._metadata = (
                self.pipeline_preprocessing.preprocess(df))

        else:
            logger.info("Using cached preprocessed data.")


        logger.info(f"x_train shape: {self._tensor_data['x_train'].shape}, y_train shape: {self._tensor_data['y_train'].shape}")
        logger.info(f"x_val shape: {self._tensor_data['x_val'].shape}, y_val shape: {self._tensor_data['y_val'].shape}")
        logger.info(f"x_test shape: {self._tensor_data['x_test'].shape}, y_test shape: {self._tensor_data['y_test'].shape}")

        return {
            'cropped_df': self._cropped_df,
            'tensor_data': self._tensor_data,
            'array_data': self._array_data,
            'scaler': self._scaler,
            'metadata': self._metadata
        }

    # TODO -> check if okay to directly use the Transformation Class or rather we go through pipeline
    def postprocess_data(self, data, selected: bool):
        """
        Postprocess (denormalize) data using the DataTransformation class.

        Args:
            data (np.ndarray): Scaled data to be denormalized.
            selected (bool): If True, denormalize only the selected feature.

        Returns:
            np.ndarray: Denormalized data.
        """
        if self._scaler is None:
            raise ValueError("Scaler is not available. Ensure preprocessing was completed first.")

        if selected:
            if 'selected_feature' not in self._metadata:
                raise ValueError("Metadata missing 'selected_feature' for denormalization.")

            selected_feature = self._metadata['selected_feature']
            logger.info("Delegating single-feature denormalization to DataTransformation.")
            return (self.data_transformation.reverse_scaler(
                data, self._scaler, selected=True, feature=selected_feature
            ))

        logger.info("Delegating full dataset denormalization to DataTransformation.")
        return self.data_transformation.reverse_scaler(data, self._scaler, selected=False)

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
            'cropped_data': self._cropped_df,
            'tensor_data': self._tensor_data,
            'array_data': self._array_data,
            'scaler': self._scaler,
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


