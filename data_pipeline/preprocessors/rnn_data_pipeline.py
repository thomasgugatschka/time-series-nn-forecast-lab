from data_pipeline.preprocessors.base_data_pipeline import BaseDataPipeline
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class RNNDataPipeline(BaseDataPipeline):
    def __init__(self, dataset_config, data_transformer, data_splitter):
        super().__init__(dataset_config=dataset_config, data_transformer=data_transformer, data_splitter=data_splitter)

        logger.info("Initializing data pipeline for RNN-based model architectures.")


    def preprocess(self, df):
        """
        Preprocessing pipeline for RNN-based models.

        Args:
            df (pd.DataFrame): Input dataframe to preprocess.

        Returns:
            # TODO !!!!!!!!!!!!!!!!!!
        """

        metadata = {}

        logger.info("Starting the preprocessing pipeline for RNN-based model architectures.")

        logger.info("Preprocessing Step: Cropping data")
        crop_start = self.dataset_config.get('crop_start')
        crop_end = self.dataset_config.get('crop_end')
        df = self.crop_data(df, crop_start, crop_end)

        logger.info("Preprocessing Step: Normalizing data")
        df, metadata_scaler = self.normalize(df, method="z_score")
        metadata.update(metadata_scaler)
        # TODO - use scaler from data_transformer rather than own implementation

        logger.info("Preprocessing Step: Calculating features")
        features = self.calculate_features()
        metadata.update(features)
        selected_feature, num_features = features['selected_feature'], features['num_features']

        logger.info("Preprocessing Step: Generating fixed time series windows")
        ts_past = self.dataset_config['ts_past']
        ts_future = self.dataset_config['ts_future']
        x_arr, y_arr = self.generate_fixed_time_series_windows(df, df.iloc[:, selected_feature], ts_past, ts_future)

        logger.info("Preprocessing Step: Splitting data")
        split_data = self.data_splitter.split(x_arr, y_arr)
        x_arr_train, x_arr_val, x_arr_test = split_data['x_arr_train'], split_data['x_arr_val'], split_data['x_arr_test']
        y_arr_train, y_arr_val, y_arr_test = split_data['y_arr_train'], split_data['y_arr_val'], split_data['y_arr_test']

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

        logger.info("Preprocessing pipeline completed for RNN-based model architectures.")

        return tensor_data, array_data, metadata

    def postprocess(self, predictions, metadata):
        """
        Postprocessing pipeline for RNN-based models.

        Args:
            predictions (pd.DataFrame): Model predictions to postprocess.
            metadata (dict): Metadata for postprocessing.

        Returns:
            pd.DataFrame: Postprocessed predictions.
        """
        # TODO !!!!!!
        df = self.denormalize(predictions, metadata, method="z_score")
        return df


