"""
CNN Segmentation Model

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSegmentationModel(nn.Module):

    # def __init__(self):
    #     super(CNNSegmentationModel, self).__init__()

    #     self.down_conv1 = nn.Conv2d(3, 32, 3)

    # def forward(self, x):
    #     x = self.down_conv1(x)

    #     return x
    def __init__(self):
        super(CNNSegmentationModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
