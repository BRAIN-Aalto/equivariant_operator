import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PolygonDataset(Dataset):
    """
    Dataset that loads MNIST images with synthetic transformations
    (rotation or translation) and samples a second target transform
    from the same source image.
    """

    def __init__(self, csv_file, split, transform_type, transform, keep_degrees=None):
        self.transform = transform
        self.split = split

        # two-axis case: source transform and target transform differ
        self.transform_type1 = transform_type
        self.transform_type2 = transform_type
        if transform_type == "shift_xy":
            self.transform_type1 = "shift_x"
            self.transform_type2 = "shift_y"

        self.folder = "mnist_combined/test" if split == "test" else "mnist_combined/train"

        df = pd.read_csv(csv_file)
        df = df[df["class"] < 9]
        df = df[df["split"] == split]

        # generate transformed file entries
        expanded_data = []
        transform_range = range(0, 360, 36) if transform_type == "rotate" else range(0, 28, 2)

        for _, row in df.iterrows():
            src_id = Path(row["file_path"]).stem
            digit = str(row["class"])

            for deg in transform_range:
                aug_name = self.get_path(src_id, deg, self.transform_type1)
                expanded_data.append(
                    {
                        "file": os.path.join(digit, aug_name),
                        "source_id": src_id,
                        "class": row["class"],
                        "degree": deg,
                    }
                )

        df = pd.DataFrame(expanded_data)

        # restrict training to specific transform magnitudes if provided
        if keep_degrees is not None:
            df = df[df["degree"].isin(keep_degrees)]
            self.keep_degrees = sorted(keep_degrees)
        else:
            self.keep_degrees = sorted(df["degree"].unique())

        # stored as: (file_path, class, degree, source_id)
        self.samples = df[["file", "class", "degree", "source_id"]].values.tolist()

    def get_path(self, fn, deg, transform_type):
        """Generate filename for a specific transform."""
        if transform_type == "rotate":
            return f"{fn}_{deg}_0_0.png"
        if transform_type == "shift_x":
            return f"{fn}_0_{deg}_0.png"
        if transform_type == "shift_y":
            return f"{fn}_0_0_{deg}.png"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls, deg, src = self.samples[idx]

        img = Image.open(Path(self.folder) / img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # randomly sample another transform of the same source image
        target_deg = random.choice(self.keep_degrees)

        target_path = Path(self.folder) / str(cls) / self.get_path(
            src, target_deg, self.transform_type2
        )

        target_img = Image.open(target_path).convert("RGB")
        if self.transform:
            target_img = self.transform(target_img)

        return img, int(deg), int(cls), target_img, target_deg