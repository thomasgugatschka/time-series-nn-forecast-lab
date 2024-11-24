import torch
import torch.nn as nn
import numpy as np
from abc import ABC

from models.base_model import BaseModel


# Positional Encoding für die Zeitdimension
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# TODO Confirm that input_size and output_size mapping is corret
# Transformer Architektur für Zeitreihen-Daten als Subklasse von BaseModel
class Transformer(BaseModel):
    def __init__(self, input_size, output_size, embed_dim, num_heads, num_layers, forward_dim, dropout):
        super(Transformer, self).__init__()

        # Embedding layer to transform input features (input_size) into the embed dimension (embed_dim)
        self.embedding = nn.Linear(input_size, embed_dim)

        # Define Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to project the final representation to the desired output size
        self.fc_out = nn.Linear(embed_dim, output_size)  # Using output_size here
        self.model_name = "Transformer"

    def forward(self, x):
        # Project input features to the embedding dimension
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # Transformer expects input in (seq_len, batch_size, embed_dim)

        # Apply the Transformer Encoder
        x = self.transformer_encoder(x)

        # Bring it back to (batch_size, seq_len, embed_dim) format
        x = x.permute(1, 0, 2)

        # Use the final time step's output for prediction and project to output size
        return self.fc_out(x[:, -1, :])  # Predict output based on the final time step
