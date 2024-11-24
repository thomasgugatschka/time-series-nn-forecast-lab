import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from abc import ABC

from models.base_model import BaseModel

# For original implementation: https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/add_test.py


# Crop Module for Padding
class Crop(nn.Module):
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()

# Temporal Causal Layer
class TemporalCausalLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalCausalLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1)

        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)

# Temporal Convolution Network - TCN Model
class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation=2, num_layers=None):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = num_layers if num_layers is not None else len(num_channels)
        dilation_rate = dilation
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }

        for i in range(num_levels):
            dilation = dilation_rate ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCausalLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# TCN Model Using BaseModel
class TCN(BaseModel):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilation, num_layers=None, model_name="TCN"):
        super(TCN, self).__init__(model_name=model_name)
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, dilation=dilation, num_layers=num_layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        # Use the last time step for prediction
        return self.linear(y[:, :, -1])
