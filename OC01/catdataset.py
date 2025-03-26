"""
Cat Dataset (CatDataset)

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

from pathlib import Path
import pickle
import random
from typing import Callable, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn


class CatDataset(Dataset):

    __slots__ = [
        'data', 'masks', 'transform', 'random_transform', 'image_size',
        'mask_channels'
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        fformat: str = 'parquet',
        transform: Optional[Callable] = None,
        image_size=[224, 224],
        image_channels: int = 3,
        mask_channels: int = 1,
        random_transform: bool = True,
    ):
        """Feral cats segmentation dataset"""
        super(CatDataset, self).__init__()

        allowed_formats = [
            'parquet',
            'pkl',
        ]

        split_selection = ['train', 'valid', 'test']

        if fformat not in allowed_formats:
            raise ValueError(
                f'Selection {fformat} is not valid. Choose from: {" | ".join(allowed_formats)}'
            )

        if split not in split_selection:
            raise ValueError(
                f'Split {split} is not valid. Choose from: {" | ".join(split_selection)}'
            )

        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f'Directory {root} does not exist.')

        if fformat == 'parquet':
            data_path = root_path / f"{fformat}/{split}_dataset.parquet"
            if not data_path.exists():
                raise ValueError(f'File {data_path} not found.')

            dataset = pl.read_parquet(data_path).to_numpy()

            split_at = image_size[0] * image_size[1] * image_channels

            # DO NOT delete formatting comments starting with 'yapf'
            # yapf: disable
            self.data = (dataset[:, :split_at]
                .astype(np.uint8)
                .reshape(-1, image_size[0], image_size[1], image_channels)
                )

            self.masks = (dataset[:, split_at:]
                .reshape(-1, image_size[0], image_size[1], mask_channels)
                )
            # yapf: enable

        elif fformat == 'pkl':
            data_path = root_path / f"{fformat}/{split}_dataset.pkl"
            if not data_path.exists():
                raise ValueError(f'File {data_path} not found.')

            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)

            images = [pickle.loads(img) for img, mask in dataset]
            masks = [pickle.loads(mask) for img, mask in dataset]

            self.data = np.stack(images).astype(np.uint8).reshape(
                -1, image_size[0], image_size[1], image_channels)

            self.masks = np.stack(masks).astype(np.uint8).reshape(
                -1, image_size[0], image_size[1], mask_channels)

        self.image_size = image_size

        self.mask_channels = mask_channels

        self.random_transform = random_transform

        self.transform = transform if transform != None else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.random_transform:
            image, mask = self._random_transform(self.data[index],
                                                 self.masks[index])

            return (
                self.transform(image),
                torch.tensor(mask, dtype=torch.long),
            )
        else:
            return (
                self.transform(self.data[index]),
                torch.tensor(self.masks[index], dtype=torch.long),
            )

    def _random_transform(self, image, mask):
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)

        # Color Jitter (Color Shift) image only
        color_jitter = transforms.ColorJitter(brightness=0.2,
                                              contrast=0.2,
                                              saturation=0.2,
                                              hue=0.1)
        image = color_jitter(image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=[192, 192])
        image = TF.resized_crop(image, i, j, h, w, self.image_size)
        mask = TF.resized_crop(mask, i, j, h, w, self.image_size)

        # Random rotation
        # Causes edge values to be in negative range after Normalization, it is normal behaviour
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        image = np.array(image)
        mask = np.array(mask).reshape(self.image_size[0], self.image_size[1],
                                      self.mask_channels)

        return image, mask


if __name__ == '__main__':
    import timeit

    root = r"D:\storage\feral-cat-segmentation.v1i.sam2"
    image_size = [224, 224]
    image_channels = 3
    mask_channels = 1

    def getds():
        return CatDataset(
            root=root,
            split='train',
            fformat='parquet',
            image_size=image_size,
            image_channels=image_channels,
            mask_channels=mask_channels,
        )

    ds = getds()

    # 5 iterations
    # parquet:  5.692191799986176    seconds    1.1384383599972352   s/per
    # pkl:      2.6891325000324287   seconds    0.5378265000064857   s/per
    tt = timeit.timeit("getds()", globals=globals(), number=5)

    data, label = ds.__getitem__(0)
    print(data, label)
    print(f"Number of data: {ds.__len__()}")
    print(f"5 iter: {tt} s\t1 iter: {tt/5}")
    """
    Visualization code was generated with GenAI
    Manually commented on functionality
    """
    import matplotlib.pyplot as plt

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

    # Select 9 random samples
    num_samples = 9
    indices = list(range(min(num_samples, len(ds))))

    # Create a 3x6 grid (image-mask pairs)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    for i, idx in enumerate(indices):
        image, mask = ds[idx * 3]

        # Unnormalize image
        image = unnormalize(image, mean, std).astype(np.uint8)

        # Convert mask to numpy
        mask = mask.squeeze().numpy()

        # Plot image
        axes[i // 3, (i % 3) * 2].imshow(image)
        axes[i // 3, (i % 3) * 2].set_title(f"Image {idx*3}")
        axes[i // 3, (i % 3) * 2].axis("off")

        # Plot mask
        axes[i // 3, (i % 3) * 2 + 1].imshow(mask, cmap="gray")
        axes[i // 3, (i % 3) * 2 + 1].set_title(f"Mask {idx*3}")
        axes[i // 3, (i % 3) * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()
