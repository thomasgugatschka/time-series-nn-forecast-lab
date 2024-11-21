import torch
import copy
from models.base_wrapper import BaseWrapper

class RNNWrapper(BaseWrapper):
    def __init__(self, model, optimizer, loss_fn, callbacks=None):
        """
        RNN Wrapper that inherits from BaseWrapper to encapsulate common functionality and handle callbacks.

        Args:
            model: The RNN model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
            callbacks (list, optional): List of callbacks to use during training.
        """
        # Call the constructor of the BaseWrapper to initialize model, optimizer, loss_fn, and callbacks
        super().__init__(model, optimizer, loss_fn, callbacks)

    # The train_epoch, validate, predict, and on_epoch_end methods are inherited from BaseWrapper.
    # If RNN-specific logic needs to be added, you can override these methods here.


    # TODO - could be removed
    def fit(self, x_train, y_train, x_val, y_val, epochs):
        training_loss = []
        validation_loss = []
        best_params = None
        min_val_loss = float('inf')

        for epoch in range(epochs):
            # Training step
            self.model.train()
            prediction = self.model(x_train)
            loss = self.loss_fn(prediction, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_prediction = self.model(x_val)
                val_loss = self.loss_fn(val_prediction, y_val)

            training_loss.append(loss.item())
            validation_loss.append(val_loss.item())

            if val_loss.item() < min_val_loss:
                best_params = copy.deepcopy(self.model.state_dict())
                min_val_loss = val_loss.item()

            # Trigger callbacks at the end of the epoch
            #for callback in self.callbacks:
            #    callback.on_epoch_end(epoch, self, x_val, y_val)

        self.model.load_state_dict(best_params)
        return training_loss, validation_loss


