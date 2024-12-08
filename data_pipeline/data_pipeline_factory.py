from data_pipeline.rnn_data_pipeline import RNNDataPipeline
from data_pipeline.transformer_data_pipeline import TransformerDataPipeline

class DataPipelineFactory:

    @staticmethod
    def get_pipeline(model_type, dataset_config, data_transformer, data_splitter):
        if model_type == "RNN" or model_type == "GRU" or model_type == "LSTM":
            return RNNDataPipeline(
                dataset_config=dataset_config,
                data_transformer=data_transformer,
                data_splitter=data_splitter)
        elif model_type == "Transformer":
            return TransformerDataPipeline(
                dataset_config=dataset_config,
                data_transformer=data_transformer,
                data_splitter=data_splitter)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
