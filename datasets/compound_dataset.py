import os
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CompoundPolygonDataset(Dataset):
    def __init__(self, csv_file, split, transform,  keep_degrees=None):
        
        self.transform = transform
        self.split = split
        self.folder = "mnist_combined/test" if split=="test" else "mnist_combined/train"

        df = pd.read_csv(csv_file)
        df = df[df["class"]<9]
        df = df[df["split"] == split]

        expanded_data = []
        
        for _, row in df.iterrows():
            path_obj = Path(row['file_path'])
            fn = path_obj.stem 
            digit = str(row['class'])
            
            aug_name = self.get_path(fn, keep_degrees[0], keep_degrees[1])
            full_path = os.path.join(digit, aug_name)
            expanded_data.append({
                "file": full_path,
                "source_id": fn,
                "class": row['class']
            })

        df = pd.DataFrame(expanded_data)


        # samples: (file, cls, deg, source_id)
        self.samples = df[["file", "class", "source_id"]].values.tolist()

    def get_path(self, fn, deg1, deg2):
        return f"{fn}_0_{deg1}_{deg2}.png"
  
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls, src = self.samples[idx]

        img = Image.open(Path(os.path.join(self.folder, img_path))).convert("RGB")
        if self.transform:
            img = self.transform(img)


        return img, int(cls)
