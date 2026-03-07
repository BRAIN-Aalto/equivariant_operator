import os
from pathlib import Path

from PIL import Image
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

"""Download MNIST and export raw grayscale PNGs under mnist_local/."""


ROOT_DATA = "mnist_data"
ROOT_OUT = "mnist_local"


def save_split(splitset, split_name: str) -> None:
    """Save one MNIST split to mnist_local/<split>/<digit>/<idx>.png."""
    split_dir = Path(ROOT_OUT) / split_name

    for digit in range(10):
        (split_dir / str(digit)).mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(splitset)), desc=f"saving {split_name}"):
        img, label = splitset[idx]
        img = transforms.ToPILImage()(img)
        out_path = split_dir / str(label) / f"{idx}.png"
        img.save(out_path)


def main() -> None:
    os.makedirs(ROOT_OUT, exist_ok=True)

    transform = transforms.ToTensor()

    trainset = datasets.MNIST(
        root=ROOT_DATA,
        train=True,
        download=True,
        transform=transform,
    )
    testset = datasets.MNIST(
        root=ROOT_DATA,
        train=False,
        download=True,
        transform=transform,
    )

    save_split(trainset, "train")
    save_split(testset, "test")


if __name__ == "__main__":
    main()
