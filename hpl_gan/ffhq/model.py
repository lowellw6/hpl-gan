import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.alexnet import AlexNet


class ConvGenerator(nn.Module):

    def __init__(self, z_size, conv_reshape):
        super().__init__()
        self.z_size = z_size
        self.conv_reshape = conv_reshape

        self.fc1 = nn.Linear(z_size, z_size)
        self.fc2 = nn.Linear(z_size, z_size)
        self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.tconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2)
        self.tconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.tconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        self.tconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(z.shape[0], *self.conv_reshape)
        x = F.leaky_relu(self.tconv1(x), 0.3)
        x = F.leaky_relu(self.tconv2(x), 0.3)
        x = F.leaky_relu(self.tconv3(x), 0.3)
        x = F.leaky_relu(self.tconv4(x), 0.3)
        x = F.leaky_relu(self.tconv5(x), 0.3)
        return self.tconv6(x)


class ConvEncoder(nn.Module):

    def __init__(self, z_size):
        super().__init__()
        self.z_size = z_size

        self.alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.features = self.alexnet.features
        self.avgpool = self.alexnet.avgpool

        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, z_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)

