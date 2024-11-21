from models.base_wrapper import BaseWrapper


class GRUWrapper(BaseWrapper):
    def __init__(self, model, optimizer, loss_fn, callbacks=None):
        """
        GRU Wrapper that inherits from BaseWrapper to encapsulate common functionality and handle callbacks.

        Args:
            model: The GRU model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
            callbacks (list, optional): List of callbacks to use during training.
        """
        # Call the constructor of the BaseWrapper to initialize model, optimizer, loss_fn, and callbacks
        super().__init__(model, optimizer, loss_fn, callbacks)

    # The train_epoch, validate, predict, and on_epoch_end methods are inherited from BaseWrapper.
    # If GRU-specific logic needs to be added, you can override these methods here.