"""
Segmentation Model
CNN, ViT

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

from pathlib import Path

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
    EPOCHS = 100
    WORKERS = 2  # number of cpu to load data
    PREFETCH_FACTOR = 4  # number of batch to prefetch
    BATCH_SIZE = 32
    LR = 0.001  # Optimizer learning rate
    WEIGHT_DECAY = 0.01  # L2 Regularization

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ROOT = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\seg"
    IMAGE_SIZE = [224, 224]
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1
    NUM_CLASSES = 2

    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='train',
            fformat='pkl',
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            mask_channels=MASK_CHANNELS,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='test',
            fformat='pkl',
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            mask_channels=MASK_CHANNELS,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=False,
    )
    print("DATASET LOADED\n")

    ### EDIT THIS PART FOR DIFFERENT MODELS ###
    model = CNNSegmentationModel()

    # criterion = CNNSegmentationModelLoss()
    # criterion = F.binary_cross_entropy_with_logits
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
    torch.backends.cudnn.benchmark = True

    model.cuda()
    print("MODEL LOADED\n")

    print("TRAINING START")
    # DO NOT delete formatting comments starting with 'yapf'
    # yapf: disable
    ACC_DIVISOR = BATCH_SIZE * (NUM_CLASSES - 1) * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    def metrics(pred, label):
        label = label.cpu().numpy().astype(np.uint8)
        pred_choice = pred.cpu().data.max(1)[1].numpy().astype(np.uint8)
        correct = np.sum(pred_choice == label)
        acc = correct / ACC_DIVISOR

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((label == l))
            total_correct_class[l] += np.sum((pred_choice == l) & (label == l))
            total_iou_deno_class[l] += np.sum(((pred_choice == l) | (label == l)))

        mIoU = np.mean(
            np.array(total_correct_class) /
            (np.array(total_iou_deno_class, dtype=float) + 1e-6))

        return acc, mIoU
    # yapf: enable

    PCW = torch.Tensor(np.ones(NUM_CLASSES)).cuda()

    best_test_mIoU = 0
    best_train_mIoU = 0
    with open(f"{SAVE_PATH}/log.csv", 'w', newline='') as csvfile:
        csvfile.write(
            f"epoch,train_loss,train_acc,train_mIoU,test_loss,test_acc,test_mIoU\n"
        )
    write_str = ""

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")

        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_train_mIoU_list = []

        model = model.train()

        for i, (data, label) in tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            data = data.cuda()
            label = label.cuda().permute(0, 3, 1, 2).squeeze(1).long()

            pred = model(data)
            loss = criterion(pred, label, PCW)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc, mIoU = metrics(pred, label)

            epoch_train_acc_list.append(acc)
            epoch_train_loss_list.append(loss.item())
            epoch_train_mIoU_list.append(mIoU)

        scheduler.step()

        train_acc = np.mean(epoch_train_acc_list)
        train_loss = np.mean(epoch_train_loss_list)
        train_mIoU = np.mean(epoch_train_mIoU_list)

        print(
            f"TRAIN | Loss: {train_loss} | Acc: {train_acc} | mIoU: {train_mIoU}"
        )

        epoch_test_loss_list = []
        epoch_test_acc_list = []
        epoch_test_mIoU_list = []

        model = model.eval()

        with torch.no_grad():
            for i, (data, label) in tqdm(enumerate(test_loader),
                                         total=len(test_loader)):
                data = data.cuda()
                label = label.cuda().permute(0, 3, 1, 2).squeeze(1).long()

                pred = model(data)
                loss = criterion(pred, label, PCW)

                acc, mIoU = metrics(pred, label)

                epoch_test_acc_list.append(acc)
                epoch_test_loss_list.append(loss.item())
                epoch_test_mIoU_list.append(mIoU)

        test_acc = np.mean(epoch_test_acc_list)
        test_loss = np.mean(epoch_test_loss_list)
        test_mIoU = np.mean(epoch_test_mIoU_list)

        print(
            f"TEST | Loss: {test_loss} | Acc: {test_acc} | mIoU: {test_mIoU}\n"
        )

        write_str += f"{epoch+1},{train_loss},{train_acc},{train_mIoU},{test_loss},{test_acc},{test_mIoU}\n"

        if test_mIoU >= best_test_mIoU:
            best_test_mIoU = test_mIoU
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}_cnn.pth")
        elif train_mIoU >= best_train_mIoU:
            best_train_mIoU = train_mIoU
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}_cnn.pth")
        elif epoch + 1 == EPOCHS:
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}_cnn.pth")

    with open(f"{SAVE_PATH}/log.csv", 'a', newline='') as csvfile:
        csvfile.write(write_str)


if __name__ == '__main__':
    main()
