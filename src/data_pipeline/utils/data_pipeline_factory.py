from src.data_pipeline.preprocessors.rnn_data_pipeline import RNNDataPipeline
from src.data_pipeline.preprocessors.transformer_data_pipeline import TransformerDataPipeline

class DataPipelineFactory:

    @staticmethod
    def create_data_pipeline(model_type, dataset_config, data_transformer, data_splitter, window_size=None):
        if (model_type == "RNN" or model_type == "GRU" or model_type == "LSTM" or
                model_type == "BiLSTM" or model_type == "TCN" or model_type == "CNN"): # RNN-based Architectures
            return RNNDataPipeline(
                dataset_config=dataset_config,
                data_transformer=data_transformer,
                data_splitter=data_splitter,
                window_size=window_size)
        elif model_type == "Transformer": # Transformer-based  Architecture
            return TransformerDataPipeline(
                dataset_config=dataset_config,
                data_transformer=data_transformer,
                data_splitter=data_splitter,
                window_size=window_size)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
