from models.rnn.rnn_model import RNN
from models.lstm.lstm_model import LSTM
from models.bilstm.bilstm_model import BiLSTM
from models.gru.gru_model import GRU
#from models.transformer import Transformer
#from models.tcn import TCN
#from models.cnn import CNN



# or create_model
def model_factory(model_type, **params):
    if model_type == 'RNN':
        return RNN(**params)
    elif model_type == 'LSTM':
        return LSTM(**params)
    elif model_type == 'BiLSTM':
        return BiLSTM(**params)
    elif model_type == 'GRU':
        return GRU(**params)
#    elif model_type == 'Transformer':
#        return Transformer(**params)
#    elif model_type == 'TCN':
#        return TCN(**params)
#    elif model_type == 'CNN':
#        return CNN(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")