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


# Reading U-Net (arXiv): https://arxiv.org/abs/1505.04597
# Reading Conv2d (Medium): https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
# Reading Conv2d (PyTorch): https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # Conv2d    (in, out, 3, padding=1)
        # BatchNorm (out)
        # ReLU
        # Conv2d    (out, out, 3, padding=1)
        # BatchNorm (out)
        # ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()

        # MaxPool2d (2)
        # ConvBlock (in, out)
        self.maxpool = nn.MaxPool2d(2)

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        # Conv2d    (in, out, 3, padding=1)
        # Upsample  (scale_factor=2, mode='bilinear', align_corners=True)
        # ConvBlock (in, out)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_conn):
        x = self.up(x)

        # concat x & skip_conn at dimension 1
        # x:            (B, C , H, W)
        # skip_conn:    (B, C , H, W)
        # result:       (B, 2C, H, W)
        x = torch.cat([x, skip_conn], dim=1)

        x = self.conv(x)

        return x


# Acc: 0.9231848684297939 | mIoU: 0.7674362504791558
class CNNSegmentationModel(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(CNNSegmentationModel, self).__init__()

        # input
        # in_channels -> 8
        self.in_conv = ConvBlock(in_channels, 6)

        # contracting (down, feature extraction)
        # 8  -> 16
        # 16 -> 32
        # 32 -> 64
        self.down_conv1 = DownConv(6, 12)
        self.down_conv2 = DownConv(12, 24)
        self.down_conv3 = DownConv(24, 48)

        # expansive (up, segmentation)
        # (64/2, 32)  = 64  -> 32
        # (32/2, 16)  = 32  -> 16
        # (16/2, 8)   = 16  -> 8
        self.up_conv1 = UpConv(48, 24)
        self.up_conv2 = UpConv(24, 12)
        self.up_conv3 = UpConv(12, 6)

        # output
        # 8 -> num_classes
        self.out_conv = nn.Sequential(
            nn.Dropout2d(p=0.4),
            ConvBlock(6, num_classes),
        )

    def forward(self, x):
        out1 = self.in_conv(x)

        out2 = self.down_conv1(out1)
        out3 = self.down_conv2(out2)
        out4 = self.down_conv3(out3)

        x = self.up_conv1(out4, out3)
        x = self.up_conv2(x, out2)
        x = self.up_conv3(x, out1)

        x = self.out_conv(x)

        return x


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
