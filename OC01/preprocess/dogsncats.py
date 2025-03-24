"""
Dogs vs. Cats Preprocessing (DNC)

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 24 March 2025
"""

# from pathlib import Path
# import pickle
# import gc

# import numpy as np
# import polars as pl
# from PIL import Image
# from tqdm import tqdm


# def preprocess(root):
#     root_path = Path(root)

#     if not root_path.exists():
#         raise (FileExistsError(f"Directory {root} does not exist."))

#     train_path = root_path / "train/train"
#     test_path = root_path / "test1/test1"

#     splits = ["train", "test"]
#     file_paths = [train_path, test_path]

#     parquet_path = f"{root_path}/parquet"
#     Path(parquet_path).mkdir(parents=True, exist_ok=True)

#     pkl_path = f"{root_path}/pkl"
#     Path(pkl_path).mkdir(parents=True, exist_ok=True)

#     for i, split in enumerate(splits):
#         for file in tqdm(list(file_paths[i].glob("*.jpg"))):
#             img = np.array(
#                 Image.open(file).convert("RGB").resize(
#                     (224, 224),
#                     resample=Image.Resampling.BILINEAR)).astype(np.uint8)

#             ### NOTE: The labels here are swapped, compared to the original cat=1, dog=0
#             ###       to be consistent with the segmentation task.
#             if file.name.find('cat') != -1:
#                 label = 1
#             else:
#                 label = 0

#             del img
#             del label
#             gc.collect()

# if __name__ == '__main__':
#     root = r"D:\storage\catsndogs"
#     preprocess(root=root)
