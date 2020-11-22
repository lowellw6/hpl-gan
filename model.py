import torch
import torch.nn as nn


class AE(nn.Module):
    """
    Adapted from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
    def __init__(self, input_shape, hidden_size=128):
        super().__init__()
        self.input_shape = input_shape

        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=hidden_size
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_size, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        return torch.relu(code)

    def decode(self, code):
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed


class LatentDiscriminatorMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.dense = nn.Linear(
            in_features=input_size, out_features=1
        )

    def forward(self, code):
        return torch.sigmoid(self.dense(code))


class LatentGeneratorMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.dense = nn.Linear(
            in_features=input_size, out_features=output_size
        )

    def forward(self, prior):
        return torch.relu(self.dense(prior))