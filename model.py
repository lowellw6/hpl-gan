import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """
    Adapted from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    """
    def __init__(self, input_shape, z_size):
        super().__init__()
        self.input_shape = input_shape

        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=z_size*2)
        
        self.encoder_output_layer = nn.Linear(in_features=z_size*2, out_features=z_size)
        
        self.decoder_hidden_layer = nn.Linear(in_features=z_size, out_features=z_size*2)
        
        self.decoder_output_layer = nn.Linear(in_features=z_size*2, out_features=input_shape)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = F.leaky_relu(activation)
        code = self.encoder_output_layer(activation)
        code = F.leaky_relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = F.leaky_relu(activation)
        activation = self.decoder_output_layer(activation)
        return torch.tanh(activation)

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = F.leaky_relu(activation)
        code = self.encoder_output_layer(activation)
        return F.leaky_relu(code)

    def decode(self, code):
        activation = self.decoder_hidden_layer(code)
        activation = F.leaky_relu(activation)
        activation = self.decoder_output_layer(activation)
        return torch.tanh(activation)


class PixelDiscriminatorMLP(nn.Module):
    """
    Adapted from: https://github.com/Garima13a/MNIST_GAN/blob/master/MNIST_GAN_Solution.ipynb
    """
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        # dropout layer 
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2) 
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.dropout(x)
        return self.fc4(x)


class PixelGeneratorMLP(nn.Module):
    """
    Adapted from: https://github.com/Garima13a/MNIST_GAN/blob/master/MNIST_GAN_Solution.ipynb
    """
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        
        # dropout layer 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.dropout(x)
        return torch.tanh(self.fc4(x))


class LatentDiscriminatorMLP(nn.Module):
    def __init__(self, z_size, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_features=z_size, out_features=hidden_dim*2)

        self.fc2 = nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim)

        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, code):
        x = F.leaky_relu(self.fc1(code))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class LatentGeneratorMLP(nn.Module):
    def __init__(self, z_size, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_features=z_size, out_features=hidden_dim)

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim*2)

        self.fc3 = nn.Linear(in_features=hidden_dim*2, out_features=z_size)

        self.dropout = nn.Dropout(0.3)

    def forward(self, code):
        x = F.leaky_relu(self.fc1(code))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)