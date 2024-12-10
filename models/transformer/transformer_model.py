

import torch
import torch.nn as nn
import numpy as np

from models.base_model import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1), :]  # Match seq_len dynamically


class Transformer(BaseModel):
    def __init__(self, input_size, output_size, embed_dim, num_heads, num_layers, forward_dim, dropout, seq_len, predict_single_feature=True):
        super(Transformer, self).__init__()

        self.model_name = 'Transformer'

        self.seq_len = seq_len
        self.output_size = output_size
        self.predict_single_feature = predict_single_feature

        # Embedding layer to project input features into embedding space
        self.embedding = nn.Linear(input_size, embed_dim)

        # Positional encoding module
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=seq_len)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(embed_dim, 1 if predict_single_feature else output_size)

    def forward(self, x):
        """
        Forward pass for the Transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, ts_future) if predict_single_feature is True,
                    otherwise (batch_size, ts_future, num_features).
        """

        # Transformer forward pass
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Apply a linear layer to project to output size
        x = self.fc_out(x)  # Shape: (batch_size, seq_len, 1) if predict_single_feature else (batch_size, seq_len, num_features)

        # Return the last `ts_future` time steps
        x = x[:, -self.output_size:, :]  # Shape: (batch_size, ts_future, 1) or (batch_size, ts_future, num_features)

        if self.predict_single_feature:
            return x.squeeze(-1)  # Shape: (batch_size, ts_future)
        return x

