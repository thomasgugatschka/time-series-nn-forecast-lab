from src.models.base_wrapper import BaseWrapper


class GRUWrapper(BaseWrapper):
    def __init__(self, model, optimizer, loss_fn, device):
        """
        GRU Wrapper that inherits from BaseWrapper to encapsulate common functionality and handle callbacks.

        Args:
            model: The GRU model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
        """

        super().__init__(model, optimizer, loss_fn, device)
