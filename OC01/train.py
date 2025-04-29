"""
Segmentation Model
CNN, ViT

CNN Models
- LeNet-1   |   MSE_LOSS        |   SGD     |   lr=0.001, weight_decay=0                    |   StepLR
- AlexNet   |   CROSS-ENTROPY   |   SGD     |   lr=0.01, momentum=0.9, weight_decay=0.0005  |   ReduceLROnPlateau
- U-Net     |   CROSS-ENTROPY   |   AdamW   |   lr=0.001, weight_decay=0.01                 |   StepLR
- VGG16     |   CROSS-ENTROPY   |   SGD     |   lr=0.01, momentum=0.9, weight_decay=0.0005  |   ReduceLROnPlateau

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

### CNN Models
from cnn.model import CNNSegmentationModel, CNNSegmentationModelLoss
from cnn.comparison_models import LeNet1, lenet1_weight_initializer, AlexNet, alexnet_weight_initializer, VGG16, vgg16_weight_initializer

### ViT (SETR) Model
from attn.model import SETR, SETRLoss

### Combined Model (Residual + SETR)
from combine.model import CombineModel, CombineModelLoss


def main():
    EPOCHS = 100  # CNN: 100 | SETR: 200
    WORKERS = 2  # number of cpu to load data
    PREFETCH_FACTOR = 4  # number of batch to prefetch
    BATCH_SIZE = 32  # CNN: 32 | SETR: 16
    LR = 0.001  # Optimizer learning rate
    WEIGHT_DECAY = 0.01  # L2 Regularization

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ROOT = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    IMAGE_SIZE = [224, 224]
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1
    NUM_CLASSES = 2

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

    ##### U-Net
    SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\unet_sm"
    model = CNNSegmentationModel(in_channels=IMAGE_CHANNELS,
                                 num_classes=NUM_CLASSES)

    # criterion = CNNSegmentationModelLoss()
    # criterion = F.binary_cross_entropy_with_logits
    criterion = F.cross_entropy
    ##### U-NET

    # ##### LeNet-1
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\lenet"
    # model = LeNet1(in_channels=3, num_classes=2)
    # model.apply(lenet1_weight_initializer)

    # criterion = F.mse_loss
    # ##### LeNet-1

    # ##### AlexNet
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\alexnet"
    # model = AlexNet(in_channels=3, num_classes=2)
    # model.apply(alexnet_weight_initializer)

    # criterion = F.cross_entropy  # multinomial logistic regression
    # ##### AlexNet

    # ##### VGG16
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\vgg"
    # model = VGG16(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES)
    # model.apply(vgg16_weight_initializer)

    # # criterion = F.binary_cross_entropy_with_logits
    # criterion = F.cross_entropy
    # ##### VGG16

    # ##### SETR
    # # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\attn\outputs\setr"
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\attn\outputs\setr2"
    # model = SETR(
    #     channels=IMAGE_CHANNELS,
    #     depth=3,
    #     dim=32,
    #     dim_head=64,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     heads=3,
    #     image_size=224,
    #     mlp_dim=64,
    #     num_classes=NUM_CLASSES,
    #     patch_size=16,
    # )

    # criterion = SETRLoss()
    # ##### SETR

    # ##### Combined
    # # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\attn\outputs\setr"
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\combine\outputs\combine_xxl"
    # model = CombineModel(
    #     channels=IMAGE_CHANNELS,
    #     depth=3,
    #     dim=32,
    #     dim_head=32,
    #     dropout=0.2,
    #     emb_dropout=0.2,
    #     heads=3,
    #     image_size=224,
    #     mlp_dim=48,
    #     num_classes=NUM_CLASSES,
    #     patch_size=16,
    #     RESF=64
    # )

    # criterion = CombineModelLoss(dim=32, RESF=64)
    # ##### Combined

    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
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
    ##### U-Net
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    ##### LeNet-1
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=LR,
    # )
    ##### AlexNet | VGG16
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=0.01,
    #     momentum=0.9,
    #     weight_decay=0.0005,
    # )
    ##### SETR
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=0.01,
    #     momentum=0.9,
    #     weight_decay=0.0005,
    # )

    ### a bit about schedulers & lr
    ### https://medium.com/data-science/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
    ##### U-Net | LeNet-1
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=20,
        gamma=0.5,
    )
    ##### AlexNet | VGG16
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,
    #     patience=10,
    #     verbose=True,
    #     threshold=0.0001,
    #     threshold_mode='rel',
    #     min_lr=1e-6,
    # )
    ##### SETR
    # scheduler = optim.lr_scheduler.PolynomialLR(
    #     optimizer,
    #     total_iters=EPOCHS,
    #     power=0.9,
    # )

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

    PCW = torch.Tensor(np.ones(NUM_CLASSES)).cuda()

    best_test_mIoU = 0
    best_train_mIoU = 0
    with open(f"{SAVE_PATH}/log.csv", 'w', newline='') as csvfile:
        csvfile.write(
            f"epoch,train_loss,train_acc,train_mIoU,test_loss,test_acc,test_mIoU\n"
        )

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")

        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_train_mIoU_list = []

        model = model.train()

        for i, (data, label) in tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            data = data.cuda()

            label = label.cuda().permute(0, 3, 1, 2).squeeze(1).long()  # cross-entropy
            # label = label.cuda().permute(0, 3, 1, 2).squeeze(1).float()  # mse_loss with SGD

            pred = model(data)

            # loss, loss4, loss3, loss2, loss1 = criterion(pred, label, PCW)

            loss = criterion(pred, label, PCW) # cross-entropy

            # ### mse_loss with SGD
            # pred_probs = torch.sigmoid(pred)
            # target = torch.zeros_like(pred_probs)
            # target[:, 0, :, :] = 1 - label
            # target[:, 1, :, :] = label
            # loss = criterion(pred_probs, target, reduction='mean')
            # ### mse_loss with SGD

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # loss1.backward(retain_graph=True)
            # loss2.backward(retain_graph=True)
            # loss3.backward(retain_graph=True)
            # loss4.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()

            acc, mIoU = metrics(pred, label)

            epoch_train_acc_list.append(acc)
            epoch_train_loss_list.append(loss.item())
            epoch_train_mIoU_list.append(mIoU)

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

                label = label.cuda().permute(0, 3, 1, 2).squeeze(1).long() # cross-entropy
                # label = label.cuda().permute(0, 3, 1, 2).squeeze(1).float()  # mse_loss with SGD

                pred = model(data)

                # loss, loss4, loss3, loss2, loss1 = criterion(pred, label, PCW)

                loss = criterion(pred, label, PCW) # cross-entropy

                # ### mse_loss with SGD
                # pred_probs = torch.sigmoid(pred)
                # target = torch.zeros_like(pred_probs)
                # target[:, 0, :, :] = 1 - label
                # target[:, 1, :, :] = label
                # loss = criterion(pred_probs, target, reduction='mean')
                # ### mse_loss with SGD

                acc, mIoU = metrics(pred, label)

                epoch_test_acc_list.append(acc)
                epoch_test_loss_list.append(loss.item())
                epoch_test_mIoU_list.append(mIoU)

        test_acc = np.mean(epoch_test_acc_list)
        test_loss = np.mean(epoch_test_loss_list)
        test_mIoU = np.mean(epoch_test_mIoU_list)

        scheduler.step() # StepLR | PolynomialLR
        # scheduler.step(test_acc) # OnPlateauReduceLR

        print(
            f"TEST | Loss: {test_loss} | Acc: {test_acc} | mIoU: {test_mIoU}\n"
        )

        with open(f"{SAVE_PATH}/log.csv", 'a', newline='') as csvfile:
            csvfile.write(f"{epoch+1},{train_loss},{train_acc},{train_mIoU},{test_loss},{test_acc},{test_mIoU}\n")

        postfix = '_cnn'
        if test_mIoU >= best_test_mIoU:
            best_test_mIoU = test_mIoU
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}{postfix}.pth")
        elif train_mIoU >= best_train_mIoU:
            best_train_mIoU = train_mIoU
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}{postfix}.pth")
        elif epoch % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}{postfix}.pth")
        elif epoch + 1 == EPOCHS:
            torch.save(model.state_dict(), f"{SAVE_PATH}/{epoch+1}{postfix}.pth")
    # yapf: enable


if __name__ == '__main__':
    main()
