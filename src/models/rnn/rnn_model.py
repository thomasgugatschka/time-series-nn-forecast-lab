import torch.nn as nn
import torch
from src.models.base_model import BaseModel

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

class RNN(BaseModel):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout, nonlinearity, **kwargs):
        super(RNN, self).__init__(model_name='RNN')

        # Validate the parameters
        self.validate_parameters(
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            nonlinearity=nonlinearity
        )

        # Initialize model parameters
        self.output_size = output_size       # The number of time steps to predict in the future
        self.input_size = input_size         # The size of the input features for each time step
        self.hidden_size = hidden_size       # The number of features in the hidden state of the RNN
        self.num_layers = num_layers         # The number of stacked RNN layers in the model
        self.dropout = dropout               # The dropout rate for regularization (between layers)
        self.nonlinearity = nonlinearity     # The activation function used in the RNN (e.g., 'tanh' or 'relu')
        self.batch_first = True              # If True, input tensors are expected to have shape (batch_size, seq_length, input_size)

        # Create RNN model
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            nonlinearity=self.nonlinearity
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

        # Propagate input through Unidirectional rnn with dropout
        output, h_n = self.rnn(x, h_0)

        # Reshape hidden states for Dense layer
        h_n = h_n.view(self.num_layers, -1, self.hidden_size)
        out = self.relu(h_n[-1])  # take the output from the last layer
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out
