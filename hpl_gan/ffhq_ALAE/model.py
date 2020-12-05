import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Latent discriminator and generator models for working with StyleALAE latent space
"""


class LatentDiscriminatorStyleALAE(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        assert z_size % 4 == 0
        self.z_size = z_size

        # Encoding layers (2 style vectors per layer)
        self.fc1 = nn.Linear(z_size, z_size)
        self.fc2 = nn.Linear(2 * z_size, z_size)
        self.fc3 = nn.Linear(2 * z_size, z_size)
        self.fc4 = nn.Linear(2 * z_size, z_size)
        self.fc5 = nn.Linear(2 * z_size, z_size)
        self.fc6 = nn.Linear(2 * z_size, z_size)
        self.fc7 = nn.Linear(2 * z_size, z_size)
        self.fc8 = nn.Linear(2 * z_size, z_size)
        self.fc9 = nn.Linear(2 * z_size, z_size)

        # Discriminator inference layers
        self.fc10 = nn.Linear(z_size, int(0.5 * z_size))
        self.fc11 = nn.Linear(int(0.5 * z_size), int(0.25 * z_size))
        self.fc12 = nn.Linear(int(0.25 * z_size), 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, styles):
        style_parts = torch.split(styles, 2, dim=1)
        s = torch.flatten(style_parts[0], start_dim=1)
        x = F.leaky_relu(self.fc1(s), 0.2)
        x = self.dropout(x)

        for i in range(2, 10):
            s = torch.flatten(style_parts[i-1], start_dim=1)
            x = torch.cat((x, s), dim=1)

            fc_layer = getattr(self, f"fc{i}")
            x = F.leaky_relu(fc_layer(x), 0.2)
            x = self.dropout(x)

        x = F.leaky_relu(self.fc10(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc11(x), 0.2)
        x = self.dropout(x)
        return self.fc12(x)


class LatentGeneratorStyleALAE(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        assert z_size % 4 == 0
        self.z_size = z_size

        # Bottlneck layers
        self.fc1 = nn.Linear(z_size, int(0.25 * z_size))
        self.fc2 = nn.Linear(int(0.25 * z_size), int(0.5 * z_size))
        self.fc3 = nn.Linear(int(0.5 * z_size), z_size)

        # Decoding layers (2 new style vectors per layer)
        self.fc4 = nn.Linear(z_size, 2 * z_size)
        self.fc5 = nn.Linear(z_size, 2 * z_size)
        self.fc6 = nn.Linear(z_size, 2 * z_size)
        self.fc7 = nn.Linear(z_size, 2 * z_size)
        self.fc8 = nn.Linear(z_size, 2 * z_size)
        self.fc9 = nn.Linear(z_size, 2 * z_size)
        self.fc10 = nn.Linear(z_size, 2 * z_size)
        self.fc11 = nn.Linear(z_size, 2 * z_size)
        self.fc12 = nn.Linear(z_size, z_size)

        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        styles = []
        for i in range(4, 12):
            fc_layer = getattr(self, f"fc{i}")
            x = fc_layer(x)

            x, s = torch.chunk(x, 2, dim=1)
            s1, s2 = torch.chunk(s, 2, dim=1)
            styles += [s1, s2]

            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

        s = self.fc12(x)
        s1, s2 = torch.chunk(s, 2, dim=1)
        styles += [s1, s2]

        return torch.stack(styles, dim=1)