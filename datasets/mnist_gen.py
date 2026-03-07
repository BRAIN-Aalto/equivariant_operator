import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from pathlib import Path
import random

"""Generate transformed RGB variants from raw MNIST PNGs."""

# Set the seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


input_root = "mnist_local"
output_root = "mnist_combined"

angles = range(0, 360, 36)
shift_list = range(0, 28, 2)
os.makedirs(output_root, exist_ok=True)
for split in ["train", "test"]:
    for d in range(10):
        os.makedirs(os.path.join(output_root, split, str(d)), exist_ok=True)

files = glob.glob(f"{input_root}/*/**/*.png", recursive=True)
files.sort()

# MAIN LOOP
for f in tqdm(files):
    parts = Path(f).parts
    digit = parts[-2]
    split =  parts[-3]
    img = Image.open(f).convert("L")
    arr = np.array(img)

    mask = (arr > 128).astype(np.uint8)
    if mask.sum() == 0:
        continue

    bg = np.random.randint(0, 2, mask.shape) * 255
    bg_rgb = np.stack([bg, bg, bg], axis=-1)

    for ang in angles:
        rot_mask = np.array(
            Image.fromarray(mask).rotate(
                ang, resample=Image.NEAREST, fillcolor=0
            )
        )

        for shift_x in shift_list:
            rolled_x = np.roll(rot_mask, shift_x, axis=1)

            for shift_y in shift_list:
                rolled_xy = np.roll(rolled_x, shift_y, axis=0)

                out = bg_rgb.copy()
                out[rolled_xy == 1] = [0, 0, 255]

                fn = os.path.basename(f).replace(".png", "")
                # Naming convention consumed by datasets/polygon_dataset.py.
                save_path = os.path.join(output_root, split, digit, f"{fn}_{ang}_{shift_x}_{shift_y}.png")

                # Skip if it exists
                if not os.path.exists(save_path):
                    Image.fromarray(out.astype(np.uint8)).save(save_path)
