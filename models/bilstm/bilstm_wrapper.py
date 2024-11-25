from models.base_wrapper import BaseWrapper

class BiLSTMWrapper(BaseWrapper):
    def __init__(self, model, optimizer, loss_fn, device):
        """
        BiLSTM Wrapper that inherits from BaseWrapper to encapsulate common functionality and handle callbacks.

        Args:
            model: The BiLSTM model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
        """

        super().__init__(model, optimizer, loss_fn, device)
