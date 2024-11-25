from models.base_wrapper import BaseWrapper

class LSTMWrapper(BaseWrapper):
    def __init__(self, model, optimizer, loss_fn, device):
        """
        LSTM Wrapper that inherits from BaseWrapper to encapsulate common functionality and handle callbacks.

        Args:
            model: The LSTM model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
        """

        super().__init__(model, optimizer, loss_fn, device)
