# uncompyle6 version 3.7.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.5.3 (default, May 15 2020, 22:04:06) 
# [GCC 8.3.0]
# Embedded file name: /media/SSD_Main/batu/vis/project/srresnet/models.py
# Compiled at: 2020-05-19 15:23:12
# Size of source mod 2**32: 17308 bytes
import torch
from torch import nn
import torchvision, math

class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvolutionalBlock, self).__init__()
        if activation is not None:
            activation = activation.lower()
            assert activation in {'leakyrelu', 'prelu', 'tanh'}
        layers = list()
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2)))
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        self.conv_block = (nn.Sequential)(*layers)

    def forward(self, input):
        output = self.conv_block(input)
        return output


class SubPixelConvolutionalBlock(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(SubPixelConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=(n_channels * scaling_factor ** 2), kernel_size=kernel_size,
          padding=(kernel_size // 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)
        return output


class ResidualBlock(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True,
          activation='PReLu')
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True,
          activation=None)

    def forward(self, input):
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual
        return output


class SRResNet(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(SRResNet, self).__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {8, 2, 4}, 'The scaling factor must be 2, 4, or 8!'
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size, batch_norm=False,
          activation='PReLu')
        self.residual_blocks = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = (nn.Sequential)(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size, batch_norm=False,
          activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output)
        return sr_imgs

class JointNet3(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_blocks=16, scaling_factor=4):
        super(JointNet3, self).__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {8, 2, 4}, 'The scaling factor must be 2, 4, or 8!'
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=64, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu', stride = 2)
        self.residual_blocks1 = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=64) for i in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=64, out_channels=64, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        
        self.conv_block3 = ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block4 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block5 = ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        
        self.conv_block6 = ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block7 = ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block8 = ConvolutionalBlock(in_channels=256, out_channels=128, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block9 = ConvolutionalBlock(in_channels=128, out_channels=64, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')
        
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor)) * 2
        self.subpixel_convolutional_blocks = (nn.Sequential)(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=64, scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])
        
        self.conv_block10 = ConvolutionalBlock(in_channels=64, out_channels=8, kernel_size=large_kernel_size,
                                              batch_norm=True, activation='Tanh')
        self.conv_block11 = ConvolutionalBlock(in_channels=8, out_channels=2, kernel_size=small_kernel_size,
                                              batch_norm=True, activation='PReLu')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs) #(64, 32, 32)
        
        residual = output
        output = self.residual_blocks1(output) #(64, 32, 32)
        output = self.conv_block2(output) #(64, 32, 32)
        output = output + residual #(64, 32, 32)
        
        output = self.conv_block3(output) #(128, 32, 32)
        output = self.conv_block4(output) #(256, 32, 32)
        output = self.conv_block5(output) #(512, 32, 32)

        output = self.conv_block6(output) #(512, 32, 32)
        output = self.conv_block7(output) #(256, 32, 32)
        output = self.conv_block8(output) #(128, 32, 32)
        output = self.conv_block9(output) #(64, 32, 32)
        
        output = self.subpixel_convolutional_blocks(output) #(64, 128, 128)
        output = self.conv_block10(output) #(8, 128, 128)
        sr_imgs = self.conv_block11(output) #(2, 128, 128)
        return sr_imgs

class JointNet2(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(JointNet, self).__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {8, 2, 4}, 'The scaling factor must be 2, 4, or 8!'
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=64, kernel_size=large_kernel_size,
          batch_norm=True,
          stride=2,
          activation='PReLu')
        self.conv_block2 = ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        self.conv_block3 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        self.conv_block4 = ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        self.residual_blocks1 = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=512) for i in range(n_blocks)])
        self.conv_block5 = ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        self.conv_block6 = ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        self.conv_block7 = ConvolutionalBlock(in_channels=256, out_channels=128, kernel_size=small_kernel_size,
          batch_norm=True,
          activation='PReLu')
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor)) * 2
        self.subpixel_convolutional_blocks = (nn.Sequential)(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=128, scaling_factor=4) for i in range(n_subpixel_convolution_blocks)])
        self.conv_block8 = ConvolutionalBlock(in_channels=128, out_channels=2, kernel_size=large_kernel_size, batch_norm=False,
          activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        residual = output
        output = self.residual_blocks1(output)
        output = self.conv_block5(output)
        output = output + residual
        output = self.conv_block6(output)
        output = self.conv_block7(output)
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block8(output)
        return sr_imgs


class JointNet(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(JointNet, self).__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {8, 2, 4}, 'The scaling factor must be 2, 4, or 8!'
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=64, kernel_size=large_kernel_size, batch_norm=False,
          activation='PReLu')
        self.residual_blocks1 = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=64) for i in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=64, out_channels=64, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.conv_block3 = ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.residual_blocks2 = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=128) for i in range(n_blocks)])
        self.conv_block4 = ConvolutionalBlock(in_channels=128, out_channels=128, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.conv_block5 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.residual_blocks3 = (nn.Sequential)(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=256) for i in range(n_blocks)])
        self.conv_block6 = ConvolutionalBlock(in_channels=256, out_channels=256, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.conv_block7 = ConvolutionalBlock(in_channels=256, out_channels=128, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        self.conv_block8 = ConvolutionalBlock(in_channels=128, out_channels=64, kernel_size=small_kernel_size,
          batch_norm=True,
          activation=None)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = (nn.Sequential)(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=64, scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])
        self.conv_block9 = ConvolutionalBlock(in_channels=64, out_channels=2, kernel_size=large_kernel_size, batch_norm=False,
          activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks1(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.conv_block3(output)
        residual = output
        output = self.residual_blocks2(output)
        output = self.conv_block4(output)
        output = output + residual
        output = self.conv_block5(output)
        residual = output
        output = self.residual_blocks3(output)
        output = self.conv_block6(output)
        output = output + residual
        output = self.conv_block7(output)
        output = self.conv_block8(output)
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block9(output)
        return sr_imgs


class Generator(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(Generator, self).__init__()
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size, n_channels=n_channels,
          n_blocks=n_blocks,
          scaling_factor=scaling_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint):
        srresnet = torch.load(srresnet_checkpoint)['model']
        self.net.load_state_dict(srresnet.state_dict())
        print('\nLoaded weights from pre-trained SRResNet.\n')

    def forward(self, lr_imgs):
        sr_imgs = self.net(lr_imgs)
        return sr_imgs


class Discriminator(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        super(Discriminator, self).__init__()
        in_channels = 3
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1 if i % 2 is 0 else 2),
              batch_norm=(i is not 0),
              activation='LeakyReLu'))
            in_channels = out_channels

        self.conv_blocks = (nn.Sequential)(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)
        return logit


class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        if not (maxpool_counter == i - 1 and conv_counter == j):
            raise AssertionError('One or both of i=%d and j=%d are not valid choices for the VGG19!' % (
             i, j))
        self.truncated_vgg19 = (nn.Sequential)(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        output = self.truncated_vgg19(input)
        return output
# okay decompiling models.cpython-37.pyc
