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
    EPOCHS = 20
    WORKERS = 2  # number of cpu to load data
    BATCH_SIZE = 32
    LR = 0.001  # Optimizer learning rate
    WEIGHT_DECAY = 0.01  # L2 Regularization

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ROOT = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    IMAGE_SIZE = [224, 224]
    IMAGE_CHANNELS = 3
    NUM_CLASSES = 1

    train_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='train',
            format='pkl',
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            mask_channels=NUM_CLASSES,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        prefetch_factor=2,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='test',
            format='pkl',
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            mask_channels=NUM_CLASSES,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        prefetch_factor=2,
        drop_last=False,
    )
    print("DATASET LOADED\n")

    ### EDIT THIS PART FOR DIFFERENT MODELS ###
    model = CNNSegmentationModel()

    # criterion = CNNSegmentationModelLoss()
    criterion = F.cross_entropy
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
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    ### a bit about schedulers & lr
    ### https://medium.com/data-science/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
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
    ACC_DIVISOR = BATCH_SIZE * NUM_CLASSES * IMAGE_SIZE[0] * IMAGE_SIZE[1]

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")

        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_train_mIoU_list = []

        model.train()

        for i, (data, label) in tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            data = data.cuda()
            label = label.cuda().permute(0, 3, 1, 2).float()

            pred = model(data)
            loss = criterion(pred, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            label = label.cpu().numpy().astype('int64')
            pred_choice = pred.cpu().data.max(1)[1].numpy().astype('int64')
            correct = np.sum(pred_choice == label)
            acc = correct / ACC_DIVISOR

            mIoU = 0
            for l in range(NUM_CLASSES):
                mIoU += np.sum((pred_choice == l) | (label == l))
            mIoU /= NUM_CLASSES

            epoch_train_acc_list.append(acc)
            epoch_train_loss_list.append(loss.item())
            epoch_train_mIoU_list.append(mIoU)

        scheduler.step()

        train_acc = np.mean(epoch_train_acc_list)
        train_loss = np.mean(epoch_train_loss_list)
        train_mIoU = np.mean(epoch_train_mIoU_list)

        print(f"TRAIN | Loss: {train_loss}\t| Acc: {train_acc}\t| mIoU: {train_mIoU}")

        epoch_test_loss_list = []
        epoch_test_acc_list = []
        epoch_test_mIoU_list = []

        model.eval()

        with torch.no_grad():
            for i, (data, label) in tqdm(enumerate(test_loader),
                                         total=len(test_loader)):
                data = data.cuda()
                label = label.cuda().permute(0, 3, 1, 2).float()

                pred = model(data)
                loss = criterion(pred, label)

                label = label.cpu().numpy().astype('int64')
                pred_choice = pred.cpu().data.max(1)[1].numpy().astype('int64')
                correct = np.sum(pred_choice == label)
                acc = correct / ACC_DIVISOR

                mIoU = 0
                for l in range(NUM_CLASSES):
                    mIoU += np.sum((pred_choice == l) | (label == l))
                mIoU /= NUM_CLASSES

                epoch_test_acc_list.append(acc)
                epoch_test_loss_list.append(loss.item())
                epoch_test_mIoU_list.append(mIoU)

        test_acc = np.mean(epoch_test_acc_list)
        test_loss = np.mean(epoch_test_loss_list)
        test_mIoU = np.mean(epoch_test_mIoU_list)

        print(f"TEST | Loss: {test_loss}\t| Acc: {test_acc}\t| mIoU: {test_mIoU}\n")


if __name__ == '__main__':
    main()
