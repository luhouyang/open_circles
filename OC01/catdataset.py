"""
Cat Dataset (CatDataset)

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 19 March 2025
"""

from pathlib import Path
import pickle
from typing import Callable, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn


class CatDataset(Dataset):

    __slots__ = ['data', 'masks', 'transform']

    def __init__(
        self,
        root: str,
        split: str = 'train',
        format: str = 'parquet',
        transform: Optional[Callable] = None,
        image_size=[224, 224],
        image_channels: int = 3,
        mask_channels: int = 1,
    ):
        """Feral cats segmentation dataset"""
        super(CatDataset, self).__init__()

        allowed_formats = [
            'parquet',
            'pkl',
        ]

        split_selection = ['train', 'valid', 'test']

        if format not in allowed_formats:
            raise ValueError(
                f'Selection {format} is not valid. Choose from: {" | ".join(allowed_formats)}'
            )

        if split not in split_selection:
            raise ValueError(
                f'Split {split} is not valid. Choose from: {" | ".join(split_selection)}'
            )

        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f'Directory {root} does not exist.')

        if format == 'parquet':
            data_path = root_path / f"{split}_dataset.parquet"
            if not data_path.exists():
                raise ValueError(f'File {data_path} not found.')

            dataset = pl.read_parquet(data_path).to_numpy()

            split_at = image_size[0] * image_size[1] * image_channels

            # DO NOT delete formatting comments starting with 'yapf'
            # yapf: disable
            self.data = (dataset[:, :split_at]
                .astype('float32')
                .reshape(-1, image_size[0], image_size[1], image_channels)
                )

            self.masks = (dataset[:, split_at:]
                .reshape(-1, image_size[0], image_size[1], mask_channels)
                )
            # yapf: enable

        elif format == 'pkl':
            data_path = root_path / f"{split}_dataset.pkl"
            if not data_path.exists():
                raise ValueError(f'File {data_path} not found.')

            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)

            images = [pickle.loads(img) for img, mask in dataset]
            masks = [pickle.loads(mask) for img, mask in dataset]

            self.data = np.stack(images).astype(np.float32)
            self.masks = np.stack(masks).astype(np.uint8)

        self.transform = transform if transform != None else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.48235,
                        0.45882,
                        0.40784,
                    ],
                    std=[
                        0.00392156862745098,
                        0.00392156862745098,
                        0.00392156862745098,
                    ],
                ),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform(self.data[index]),
            torch.tensor(self.masks[index], dtype=torch.long),
        )


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
            format='pkl',
            image_size=image_size,
            image_channels=image_channels,
            mask_channels=mask_channels,
        )

    # 5 iterations
    # parquet:  6.467789999995148   seconds     1.2935579999990297  s/per
    # pkl:      3.530102299991995   seconds     0.706020459998399   s/per
    tt = timeit.timeit("getds()", globals=globals(), number=5)
    print(f"5 iter: {tt} s\t1 iter: {tt/5}")

    ds = getds()

    print(ds.__len__())
    
    """Visualization code was generated with GenAI"""
    import matplotlib.pyplot as plt

    # Unnormalize function
    def unnormalize(image, mean, std):
        image = image.numpy().transpose(1, 2, 0)
        image = (image * np.array(std)) + np.array(mean)
        image = np.clip(image, 0, 255)
        return image

    # Normalization parameters (same as in dataset)
    mean = [0.48235, 0.45882, 0.40784]
    std = [0.00392156862745098] * 3  # Same std for all channels

    # Select 9 random samples
    num_samples = 9
    indices = list(range(min(num_samples, len(ds))))

    # Create a 3x6 grid (image-mask pairs)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    for i, idx in enumerate(indices):
        image, mask = ds[idx * 3]

        # Unnormalize image
        image = unnormalize(image, mean, std).astype('int64')

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
