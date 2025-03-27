"""
Dogs vs. Cats Dataset (DNCDataset)

Code was generated with GenAI, this script is not used in the workshop

GitHub: https://github.com/luhouyang/open_circles.git
date: 24 March 2025
"""

from pathlib import Path
from typing import Callable, Optional, List, Union, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CatDogDataset(Dataset):
    """Memory-efficient dataset for cats and dogs classification that loads images on-demand."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache_to_memory: bool = False,
        image_size: List[int] = [224, 224],
    ):
        """
        Dogs vs Cats classification dataset
        
        Args:
            root: Root directory containing the dataset 
            split: Data split ('train' or 'test')
            transform: Optional transforms to apply to images
            cache_to_memory: Whether to cache images in memory (faster but uses more RAM)
            image_size: Size to resize images to [height, width]
        """
        super(CatDogDataset, self).__init__()

        split_selection = ['train', 'test']
        if split not in split_selection:
            raise ValueError(
                f'Split {split} is not valid. Choose from: {" | ".join(split_selection)}'
            )

        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f'Directory {root} does not exist.')

        # Set up file paths based on split
        if split == 'train':
            data_path = root_path / "train/train"
        else:  # test
            data_path = root_path / "test1/test1"

        if not data_path.exists():
            raise ValueError(f'Directory {data_path} not found.')

        # Get all image files
        self.image_files = sorted(list(data_path.glob("*.jpg")))
        if not self.image_files:
            raise ValueError(f'No JPG images found in {data_path}')

        # Store dataset parameters
        self.transform = transform if transform is not None else transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        self.image_size = image_size
        self.cache_to_memory = cache_to_memory
        self.split = split

        # Cache for storing loaded images if cache_to_memory is True
        self.cache: Dict[int, Union[np.ndarray, None]] = {}

        # Pre-compute labels to avoid string operations during __getitem__
        self.labels = []
        for file in self.image_files:
            # NOTE: The labels here are swapped, compared to the original cat=1, dog=0
            # to be consistent with the segmentation task
            if file.name.find('cat') != -1:
                self.labels.append(1)
            else:
                self.labels.append(0)

        # Optionally preload all images into memory
        if self.cache_to_memory:
            print(f"Preloading {len(self.image_files)} images into memory...")
            for i in tqdm(range(len(self.image_files))):
                img = Image.open(self.image_files[i]).convert("RGB")
                img = np.array(img)
                self.cache[i] = img
            print("Preloading complete.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Get label
        label = self.labels[index]

        # Get image
        if self.cache_to_memory and index in self.cache:
            # Load from cache if available
            img_array = self.cache[index]
            img = Image.fromarray(img_array)
        else:
            # Load from file
            img = Image.open(self.image_files[index]).convert("RGB")

            # Cache the image if needed
            if self.cache_to_memory:
                self.cache[index] = np.array(img)

        # Apply transformations
        img_tensor = self.transform(img)

        return img_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # Dataset path
    root = r"D:\storage\catsndogs"

    # Create dataset
    ds = CatDogDataset(
        root=root,
        split='train',
        cache_to_memory=False,
    )

    # Timing test
    start_time = time.time()

    # First access (cold start)
    first_item = ds[0]
    first_access_time = time.time() - start_time

    # Second access (potentially cached)
    start_time = time.time()
    first_item_again = ds[0]
    second_access_time = time.time() - start_time

    print(f"Dataset size: {len(ds)} images")
    print(f"First access time: {first_access_time:.6f} seconds")
    print(f"Second access time: {second_access_time:.6f} seconds")

    # Visualization function
    def unnormalize(image,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
        image = image.numpy().transpose(1, 2, 0)
        image = ((image * np.array(std)) + np.array(mean)) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    # Visualize a few samples
    num_samples = min(4, len(ds))
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))

    for i in range(num_samples):
        image, label = ds[np.random.randint(0, len(ds))]

        # Unnormalize image for display
        img_display = unnormalize(image)

        # Plot
        axes[i].imshow(img_display)
        axes[i].set_title(f"{'Cat' if label == 1 else 'Dog'}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # DataLoader example
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )

    # Test batch loading
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")

        # Only print first batch info
        if batch_idx == 0:
            print(f"Image tensor shape: {images.shape}")
            print(f"Label tensor shape: {labels.shape}")
            print(f"Labels in batch: {labels.tolist()}")
            break
