import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.layer3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.layer4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.layer5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.layer6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x): 
        x = F.leaky_relu(self.bn1(self.layer1(x)),inplace=True)
        x = F.leaky_relu(self.bn2(self.layer2(x)),inplace=True)
        x = F.leaky_relu(self.bn3(self.layer3(x)),inplace=True)
        x = F.leaky_relu(self.bn4(self.layer4(x)),inplace=True)
        x = F.leaky_relu(self.bn5(self.layer5(x)),inplace=True)
        x = F.leaky_relu(self.bn6(self.layer6(x)),inplace=True)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # ---- Main Track
        self.layer1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        # interpolate 8x8
        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # interpolate 16x16
        self.layer3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # interpolate 32x32
        self.layer4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # interpolate 64x64
        self.layer5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        # interpolate 128x128
        self.layer6 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        # interpolate 256x256
        self.layer7 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(8)
        self.layer8 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        x = F.leaky_relu(self.bn1(self.layer1(x)),inplace=True)
        x = F.interpolate(x, size=(8,8))
        x = F.leaky_relu(self.bn2(self.layer2(x)),inplace=True)
        x = F.interpolate(x, size=(16,16))
        x = F.leaky_relu(self.bn3(self.layer3(x)),inplace=True)
        x = F.interpolate(x, size=(32,32))
        x = F.leaky_relu(self.bn4(self.layer4(x)),inplace=True)
        x = F.interpolate(x, size=(64,64))
        x = F.leaky_relu(self.bn5(self.layer5(x)),inplace=True)
        x = F.interpolate(x, size=(128,128))
        x = F.leaky_relu(self.bn6(self.layer6(x)),inplace=True)
        x = F.interpolate(x, size=(256,256))
        x = F.leaky_relu(self.bn7(self.layer7(x)),inplace=True)
        x = torch.tanh(self.layer8(x)) * 127.0
        return x

class FullNetwork(nn.Module):
    def __init__(self):
        super(FullNetwork, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x