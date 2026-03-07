import os
import random

import pandas as pd

"""Create mnist_dataset_split.csv from mnist_local train/test folders."""


random.seed(42)

"""
Scans root_dir for 'train' and 'test' folders, randomly picks a validation
set from the train folder, and writes a CSV with paths and splits.
"""
root_dir = "mnist_local"
output_csv = "mnist_dataset_split.csv"
val_ratio = 0.2
data_records = []

# 1. Process train/val split
train_root = os.path.join(root_dir, "train")

for cls in range(10):
    cls = str(cls)
    cls_dir = os.path.join(train_root, cls)
    files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])

    # Split at source image level to keep train/val disjoint.
    random.shuffle(files)

    val_count = int(len(files) * val_ratio)
    val_files = files[:val_count]
    train_files = files[val_count:]

    for f in train_files:
        data_records.append(
            {
                "file_path": os.path.join("train", cls, f),
                "class": cls,
                "split": "train",
            }
        )

    for f in val_files:
        data_records.append(
            {
                "file_path": os.path.join("train", cls, f),
                "class": cls,
                "split": "val",
            }
        )

# 2. Process test split
test_root = os.path.join(root_dir, "test")
for cls in range(10):
    cls = str(cls)
    cls_dir = os.path.join(test_root, cls)
    if os.path.exists(cls_dir):
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        for f in files:
            data_records.append(
                {
                    "file_path": os.path.join("test", cls, f),
                    "class": cls,
                    "split": "test",
                }
            )

# 3. Create dataframe and save
df = pd.DataFrame(data_records)
df["class"] = df["class"].astype(int)
df.to_csv(output_csv, index=False)
print(f"Successfully saved {len(df)} records to {output_csv}")
