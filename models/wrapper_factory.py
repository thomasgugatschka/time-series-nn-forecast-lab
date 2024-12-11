from models.rnn.rnn_wrapper import RNNWrapper
from models.lstm.lstm_wrapper import LSTMWrapper
from models.bilstm.bilstm_wrapper import BiLSTMWrapper
from models.gru.gru_wrapper import GRUWrapper
from models.transformer.transformer_wrapper import TransformerWrapper
from models.tcn.tcn_wrapper import TCNWrapper
from models.cnn.cnn_wrapper import CNNWrapper


class WrapperFactory:
    @staticmethod
    def create_wrapper(model_type, model, optimizer, loss_fn, device):
        if model_type == 'RNN':
            return RNNWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'LSTM':
             return LSTMWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'BiLSTM':
            return BiLSTMWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'GRU':
            return GRUWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'Transformer':
            return TransformerWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'TCN':
            return TCNWrapper(model, optimizer, loss_fn, device)
        elif model_type == 'CNN':
            return CNNWrapper(model, optimizer, loss_fn, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")