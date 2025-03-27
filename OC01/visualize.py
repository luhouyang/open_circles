from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from cnn.comparison_models import LeNet1, AlexNet, VGG16
from catdataset import CatDataset
from cnn.model import CNNSegmentationModel


# Unnormalize function
# 1. Convert [Tensor] to [np.array]
# 2. Transpose image shape from (c, h, w) -> (h, w, c)
# 3. Unnormalize image, p = (x * σ) + μ
# 4. Multiply by 255.0 to reverse the transforms.ToTensor() operation
# 5. Clip pixel values to between 0, 255
def unnormalize(image, mean, std):
    image = image.numpy().transpose(1, 2, 0)
    # NOTE: Normalize function
    # Reading (Normalization): https://medium.com/@piyushkashyap045/image-normalization-in-pytorch-from-tensor-conversion-to-scaling-3951b6337bc8
    # Reading (VGG16 Transforms): https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16:~:text=Finally%20the%20values%20are%20first%20rescaled%20to%20%5B0.0%2C%201.0%5D%20and%20then%20normalized%20using%20mean%3D%5B0.485%2C%200.456%2C%200.406%5D%20and%20std%3D%5B0.229%2C%200.224%2C%200.225%5D.
    # x = (p - μ) / σ
    # To reverse this operation, inverse the function
    # p = (x * σ) + μ
    image = ((image * np.array(std)) + np.array(mean)) * 255.0
    image = np.clip(image, 0, 255)
    return image


# Normalization parameters, recommended by PyTorch
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def main():
    WORKERS = 2  # number of cpu to load data
    PREFETCH_FACTOR = 4  # number of batch to prefetch
    BATCH_SIZE = 32

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ROOT = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\unet"
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\lenet"
    SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\alexnet"
    # SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs\vgg"
    MODEL_NAME = r"100_cnn.pth"
    IMAGE_SIZE = [224, 224]
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1
    NUM_CLASSES = 2

    val_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='valid',
            fformat='pkl',
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            mask_channels=MASK_CHANNELS,
            random_transform=False,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=True,
    )

    ### a bit about loading models
    ### Reading: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # model = CNNSegmentationModel(in_channels=IMAGE_CHANNELS,
    #                              num_classes=NUM_CLASSES)
    # model = LeNet1(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES)
    model = AlexNet(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES)
    # model = VGG16(in_channels=IMAGE_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(f"{SAVE_PATH}/{MODEL_NAME}", weights_only=True))
    model = model.eval()

    # DO NOT delete formatting comments starting with 'yapf'
    # yapf: disable
    ACC_DIVISOR = BATCH_SIZE * (NUM_CLASSES - 1) * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    def metrics(pred, label):
        label = label.cpu().numpy().astype(np.uint8)
        pred_choice = pred.cpu().data.max(1)[1].numpy().astype(np.uint8).reshape(-1, 224, 224, 1)
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

    preds = []
    labels = []
    images = []

    acc_list = []
    mIoU_list = []

    for i, (data, label) in enumerate(tqdm(val_loader)):
        pred = model(data)
        pred_choice = pred.cpu().data.max(1)[1].numpy().astype('int64')

        preds.append(pred_choice)
        labels.append(label)
        images.append(data)

        acc, mIoU = metrics(pred, label)

        acc_list.append(acc)
        mIoU_list.append(mIoU)

    acc = np.mean(acc_list)
    mIoU = np.mean(mIoU_list)

    print(f"TRAIN | Acc: {acc} | mIoU: {mIoU}")

    with open(f"{SAVE_PATH}/val.txt", 'w') as f:
        f.write(f"val_acc,val_mIoU\n{acc},{mIoU}")
        f.close()

    # Create a 3x6 grid (image-mask pairs)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    for i in range(6):
        pred, mask = preds[0][i], labels[0][i]

        # Unnormalize image
        image = unnormalize(images[0][i], mean, std).astype(np.uint8)

        # Convert mask to numpy
        mask = mask.squeeze().numpy()

        # Plot image
        axes[i // 2, (i % 2) * 3].imshow(image)
        axes[i // 2, (i % 2) * 3].set_title(f"Image {i}")
        axes[i // 2, (i % 2) * 3].axis("off")

        # Plot mask
        axes[i // 2, (i % 2) * 3 + 1].imshow(mask, cmap="gray")
        axes[i // 2, (i % 2) * 3 + 1].set_title(f"Mask {i}")
        axes[i // 2, (i % 2) * 3 + 1].axis("off")

        # Plot pred
        axes[i // 2, (i % 2) * 3 + 2].imshow(pred, cmap="viridis")
        axes[i // 2, (i % 2) * 3 + 2].set_title(f"Seg {i}")
        axes[i // 2, (i % 2) * 3 + 2].axis("off")

    plt.savefig(f"{SAVE_PATH}/val.png")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
