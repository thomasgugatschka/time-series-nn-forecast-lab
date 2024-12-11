from models.rnn.rnn_model import RNN
from models.lstm.lstm_model import LSTM
from models.bilstm.bilstm_model import BiLSTM
from models.gru.gru_model import GRU
from models.transformer.transformer_model import Transformer
from models.tcn.tcn_model import TCN
from models.cnn.cnn_model import CNN

from utilities.logging_utils import get_logger

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