import torch
import torch.nn as nn
from src.models.base_model import BaseModel

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

class CNN(BaseModel):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.0, num_layers=3, **kwargs):
        """
        A Convolutional Neural Network (CNN) for time series prediction.

        Args:
            input_size (int): Number of input features per time step.
            output_size (int): Number of time steps to predict as output.
            num_channels (list): List of output channels for each CNN layer.
            kernel_size (int): Size of the convolutional kernel.
            dropout (float): Dropout rate for regularization.
            num_layers (int): Number of CNN layers.
            **kwargs: Additional keyword arguments.
        """
        super(CNN, self).__init__(model_name="CNN")

        # Validate parameters
        self.validate_parameters(
            input_size=input_size,
            output_size=output_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_layers=num_layers
        )

        # Ensure num_channels matches num_layers
        if len(num_channels) < num_layers:
            num_channels = num_channels + [num_channels[-1]] * (num_layers - len(num_channels))

        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_layers = num_layers

        # Build CNN layers
        layers = []
        for i in range(self.num_layers):
            in_channels = self.input_size if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=(self.kernel_size // 2)  # Keep the sequence length constant
                )
            )
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(self.num_channels[-1], self.output_size)  # Fully connected layer for prediction

        if kwargs:
            logger.debug(f"Unused keyword arguments: {kwargs}")

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Transpose input to match Conv1d expected shape: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        # Pass through CNN layers
        x = self.cnn(x)
        # Use the last time step's features for prediction
        x = x[:, :, -1]  # Shape: (batch_size, num_channels[-1])
        # Pass through the fully connected layer
        return self.fc(x)
