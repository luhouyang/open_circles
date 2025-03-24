from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from catdataset import CatDataset
from cnn.model import CNNSegmentationModel


def main():
    WORKERS = 2  # number of cpu to load data
    PREFETCH_FACTOR = 4  # number of batch to prefetch
    BATCH_SIZE = 32

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ROOT = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    SAVE_PATH = r"C:\Users\User\Desktop\Python\open_circles\OC01\cnn\outputs"
    MODEL_NAME = r"100_cnn.pth"
    IMAGE_SIZE = [224, 224]
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1

    val_loader = DataLoader(
        dataset=CatDataset(
            root=ROOT,
            split='valid',
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

    ### a bit about loading models
    ### Reading: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model = CNNSegmentationModel()
    model.load_state_dict(
        torch.load(f"{SAVE_PATH}/{MODEL_NAME}", weights_only=True))
    model.eval()

    preds = []
    labels = []

    for i, (data, label) in enumerate(tqdm(val_loader)):
        pred = model(data)
        pred_choice = pred.cpu().data.max(1)[1].numpy().astype('int64')

        preds.append(pred_choice)
        labels.append(label)

    # Create a 3x6 grid (image-mask pairs)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    for i in range(9):
        image, mask = preds[0][i], labels[0][i]

        # Convert mask to numpy
        mask = mask.squeeze().numpy()

        # Plot image
        axes[i // 3, (i % 3) * 2].imshow(image)
        axes[i // 3, (i % 3) * 2].set_title(f"Image {i}")
        axes[i // 3, (i % 3) * 2].axis("off")

        # Plot mask
        axes[i // 3, (i % 3) * 2 + 1].imshow(mask, cmap="gray")
        axes[i // 3, (i % 3) * 2 + 1].set_title(f"Mask {i}")
        axes[i // 3, (i % 3) * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
