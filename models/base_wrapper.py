import torch

class BaseWrapper:
    def __init__(self, model, optimizer, loss_fn, device):
        """
        Base wrapper for models to encapsulate common functionality.

        Args:
            model: The neural network model instance.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for training.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, x_train, y_train):
        """
        Train the model for one epoch.

        Args:
            x_train: Training input data.
            y_train: Training target data.

        Returns:
            float: Training loss for the epoch.
        """
        self.model.train()  # Set model to training mode
        self.optimizer.zero_grad()  # Clear previous gradients
        prediction = self.model(x_train)  # Forward pass
        loss = self.loss_fn(prediction, y_train)  # Compute loss
        loss.backward()  # Backward pass to compute gradients
        self.optimizer.step()  # Update parameters
        return loss.item()

    def validate(self, x_val, y_val):
        """
        Validate the model on the validation dataset.

        Args:
            x_val: Validation input data.
            y_val: Validation target data.

        Returns:
            float: Validation loss.
        """
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            prediction = self.model(x_val)  # Forward pass
            val_loss = self.loss_fn(prediction, y_val)  # Compute loss
        return val_loss.item()

    def predict(self, x):
        """
        Make predictions using the model.

        Args:
            x: Input data for making predictions.

        Returns:
            torch.Tensor: Predicted output.
        """
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            return self.model(x)

    def get_model_name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model.get_model_name()
