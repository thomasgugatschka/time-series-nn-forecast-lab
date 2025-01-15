import torch.nn as nn
import torch
from src.models.base_model import BaseModel

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class LSTM(BaseModel):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout, **kwargs):
        super().__init__(model_name='LSTM')

        # Validate the parameters
        self.validate_parameters(
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Initialize model parameters
        self.output_size = output_size       # The number of time steps or features to predict as output
        self.num_layers = num_layers         # The number of stacked LSTM layers
        self.input_size = input_size         # The dimensionality of input features for each time step
        self.hidden_size = hidden_size       # The number of features in the hidden state of the LSTM
        self.dropout = dropout               # The dropout rate applied between LSTM layers for regularization
        self.batch_first = True              # Ensures input tensors are in the shape (batch_size, seq_length, input_size)


        # Unidirectional LSTM model with dropout
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
        )

        # Fully connected layers
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.fc_2 = nn.Linear(128, self.output_size)
        self.relu = nn.ReLU()

        if kwargs:
            logger.debug(f"Unused keyword arguments: {kwargs}")

    def forward(self, x):
        # Initialize hidden state for each layer and batch
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through Unidirectional LSTM with dropout
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Reshape hidden states for Dense layer
        hn = hn.view(self.num_layers, -1, self.hidden_size)
        out = self.relu(hn[-1])  # take the output from the last layer
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out

