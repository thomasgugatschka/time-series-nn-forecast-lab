import torch.nn as nn
import torch

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

class RNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout=0.2):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.model_name = 'rnn'

        # Unidirectional rnn model with dropout
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            nonlinearity='tanh'  # You can change this to 'relu' if needed
                                # TODO check if makes sense with relu -> another implementation / activation function
                                # TODO -> have them in config file
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

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

    def get_model_name(self):
        return self.model_name