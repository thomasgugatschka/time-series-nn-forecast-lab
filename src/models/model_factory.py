from src.models.rnn.rnn_model import RNN
from src.models.lstm.lstm_model import LSTM
from src.models.bilstm.bilstm_model import BiLSTM
from src.models.gru.gru_model import GRU
from src.models.transformer.transformer_model import Transformer
from src.models.tcn.tcn_model import TCN
from src.models.cnn.cnn_model import CNN

from src.utilities.logging_utils import get_logger

logger = get_logger(__name__)

class ModelFactory:
    @staticmethod
    def create_model(model_type, **params):
        logger.info(f"Creating model of type: {model_type}")
        logger.info("Parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        if model_type == 'RNN':
            return RNN(**params)
        elif model_type == 'LSTM':
            return LSTM(**params)
        elif model_type == 'BiLSTM':
            return BiLSTM(**params)
        elif model_type == 'GRU':
            return GRU(**params)
        elif model_type == 'Transformer':
            return Transformer(**params)
        elif model_type == 'TCN':
            return TCN(**params)
        elif model_type == 'CNN':
            return CNN(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")