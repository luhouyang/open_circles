"""
CNN Segmentation Model

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

import numpy as np
from tqdm import tqdm
import torchinfo

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from catdataset import CatDataset
from cnn.model import CNNSegmentationModel, CNNSegmentationModelLoss


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    root = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    image_size = [224, 224]
    image_channels = 3
    mask_channels = 1

    train_loader = DataLoader(
        dataset=CatDataset(
            root=root,
            split='train',
            format='pkl',
            image_size=image_size,
            image_channels=image_channels,
            mask_channels=mask_channels,
        ),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=CatDataset(
            root=root,
            split='test',
            format='pkl',
            image_size=image_size,
            image_channels=image_channels,
            mask_channels=mask_channels,
        ),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )
    print("DATASET LOADED\n")

    ### EDIT THIS PART FOR DIFFERENT MODELS ###
    model = CNNSegmentationModel()

    # criterion = CNNSegmentationModelLoss()
    criterion = F.nll_loss
    ### EDIT THIS PART FOR DIFFERENT MODELS ###

    ### a bit about Adam & AdamW
    ### https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch

    ### a bit about regularization
    ### https://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=0.001,
    #     weight_decay=0.005,
    #     betas=(0.9, 0.999),
    # )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=10,
        gamma=0.1,
    )

    torchinfo.summary(model)
    torch.compile(model)
    torch.backends.cudnn.benchmark = True

    model.cuda()
    print("MODEL LOADED\n")

    print("TRAINING START")


if __name__ == '__main__':
    main()
