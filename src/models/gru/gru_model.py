import torch.nn as nn
import torch
from src.models.base_model import BaseModel

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

class GRU(BaseModel):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout, **kwargs):
        super(GRU, self).__init__(model_name='GRU')

        # Validate the parameters
        self.validate_parameters(
            output_size=output_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

#        self.output_size = output_size
#        self.num_layers = num_layers
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.dropout = dropout
#        self.batch_first = True

    # TODO -> fix somewhere else.... casting here should not be necessary???? better fix it while reading the CONFIG
        # TODO -> should not be necessary to cast to int here... fix it while reading the CONFIG
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dropout = dropout
        self.batch_first = True

# TODO -> CHECK ALL MODELS if one initialized that I use *.self

        # Unidirectional GRU model with dropout
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
        )

        # Fully connected layers
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, self.output_size)

        if kwargs:
            logger.debug(f"Unused keyword arguments: {kwargs}")

    def forward(self, x):
        # Initialize hidden state for each layer and batch
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Propagate input through Unidirectional GRU with dropout
        output, h_n = self.gru(x, h_0)

        # Reshape hidden states for Dense layer
        h_n = h_n.view(self.num_layers, -1, self.hidden_size)
        out = self.relu(h_n[-1])  # Take the output from the last layer
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out
