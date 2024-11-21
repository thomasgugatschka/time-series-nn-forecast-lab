import torch.nn as nn
import torch
from models.base_model import BaseModel

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class LSTM(BaseModel):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.model_name = 'LSTM'

        # Unidirectional LSTM model with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

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

