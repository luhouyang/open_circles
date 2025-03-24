"""
CNN Segmentation Model
U-Net

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reading Conv2d (Medium): https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
# Reading Conv2d (PyTorch): https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#
# Reading BatchNorm (TDS): https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338/
#
# Size of output image can be calculated with the formula from (PyTorch)
# P: padding
# D: dialation
# K: kernel size
# S: stride
# H_out = [H_in + 2*P - D(K - 1) - 1] / S + 1
# W_out = [W_in + 2*P - D(K - 1) - 1] / S + 1
#
# Example, input image of (224, 224, 3)
# K = 3, P = 1, D = 1, S = 1
# H_out = [224 + 2*1 - 1(3 - 1) - 1] / 1 + 1
#       = 224 + 2 - 2 - 1 + 1
#       = 224
class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # MaxPool2d (2)
        # Conv2d    (in, out, 3, padding=1)
        # BatchNorm (out)
        # ReLU
        # Conv2d    (out, out, 3, padding=1)
        # BatchNorm (out)
        # ReLU
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CNNSegmentationModel(nn.Module):

    def __init__(self):
        super(CNNSegmentationModel, self).__init__()

        self.down_conv1 = nn.Conv2d(3, 32, 3)

    def forward(self, x):
        x = self.down_conv1(x)

        return x

    # def __init__(self):
    #     super(CNNSegmentationModel, self).__init__()
    #     # Encoder
    #     self.encoder1 = nn.Sequential(
    #         nn.Conv2d(3, 32, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(32),
    #         nn.MaxPool2d(2, 2),
    #     )

    #     self.encoder2 = nn.Sequential(
    #         nn.Conv2d(32, 64, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(64),
    #         nn.Conv2d(64, 128, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(128),
    #         nn.MaxPool2d(2, 2)
    #     )

    #     self.encoder3 = nn.Sequential(
    #         nn.Conv2d(128, 256, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(256),
    #         nn.MaxPool2d(2, 2)
    #     )

    #     # Decoder
    #     self.decoder1 = nn.Sequential(
    #         nn.Conv2d(256, 128, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(128),
    #         nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
    #     )

    #     self.decoder2 = nn.Sequential(
    #         nn.Conv2d(256, 64, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(64),
    #         nn.Conv2d(64, 32, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(32),
    #         nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
    #     )

    #     self.decoder3 = nn.Sequential(
    #         nn.Conv2d(64, 32, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(32),
    #         nn.Conv2d(32, 16, 3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1)
    #     )

    # def forward(self, x):
    #     out1 = self.encoder1(x)
    #     out2 = self.encoder2(out1)
    #     x = self.encoder3(out2)

    #     x = self.decoder1(x)
    #     x = torch.cat([x, out2], dim=1)
    #     x = self.decoder2(x)
    #     x = torch.cat([x, out1], dim=1)
    #     x = self.decoder3(x)
    #     return x


class CNNSegmentationModelLoss(nn.Module):

    __slots__ = ['l2_reg_scale']

    def __init__(self, l2_reg_scale=0.01):
        super(CNNSegmentationModelLoss, self).__init__()

        self.l2_reg_scale = l2_reg_scale

    def forward(self, pred, label, mat):
        loss = F.nll_loss(pred, label)
        reg_loss = mat**2 / 2

        total_loss = loss + reg_loss * self.l2_reg_scale

        return total_loss
