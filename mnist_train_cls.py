import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import PolygonDataset
from models import LATENT_DIM, LinearClassifier
from training import rotate, xshift, yshift, xyshift
from training.base import training_loop


TRANSFORM_CONFIG = {
    "rotate": {
        "trained_degrees": [288, 324, 0, 36, 72],
        "block_size": 10,
        "num_stages": 1,
        "trainer": rotate,
    },
    "shift_x": {
        "trained_degrees": [24, 26, 0, 2, 4],
        "block_size": 14,
        "num_stages": 1,
        "trainer": xshift,
    },
    "shift_y": {
        "trained_degrees": [24, 26, 0, 2, 4],
        "block_size": 14,
        "num_stages": 1,
        "trainer": yshift,
    },
    "shift_xy": {
        "trained_degrees": list(range(0, 28, 2)),
        "block_size": 14,
        "num_stages": 2,
        "trainer": xyshift,
    },
}


def parse_degrees(raw):
    if raw is None:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_dataset(csv_path, transform_type, trained_degrees):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = PolygonDataset(
        csv_path,
        split="train",
        transform=transform,
        transform_type=transform_type,
        keep_degrees=trained_degrees,
    )

    val_set = PolygonDataset(
        csv_path,
        split="val",
        transform=transform,
        transform_type=transform_type,
        keep_degrees=trained_degrees,
    )

    return train_set, val_set


def build_dataloaders(train_set, val_set, batch_size, num_workers, pin_memory, prefetch_factor):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_model(mode, device, block_size, num_stages):
    kwargs = {
        "num_classes": 9,
        "num_stages": num_stages,
        "device": device,
    }

    if mode == "no_op":
        return LinearClassifier(**kwargs).to(device)

    if mode == "fixed_op":
        return LinearClassifier(block_size=block_size, **kwargs).to(device)

    if mode == "learned_op":
        return LinearClassifier(block_size=LATENT_DIM, learnable_P=True, **kwargs).to(device)

    raise ValueError(f"Unknown mode: {mode}")


def run_experiment(args):
    cfg = TRANSFORM_CONFIG[args.transform]
    trained_degrees = parse_degrees(args.trained_degrees) or cfg["trained_degrees"]

    train_set, val_set = build_dataset(args.csv, args.transform, trained_degrees)
    train_loader, val_loader = build_dataloaders(
        train_set,
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    print(f"\\n===== Training {args.transform} =====")
    print(f"device={args.device} batch_size={args.batch_size} epochs={args.epochs} lr={args.lr}")

    for mode in args.modes:
        model = build_model(
            mode=mode,
            device=args.device,
            block_size=cfg["block_size"],
            num_stages=cfg["num_stages"],
        )

        save_dir = os.path.join(args.checkpoint_root, f"mnist_{args.transform}_cls_{mode}")

        if args.transform == "shift_xy":
            xyshift.training_loop(
                model,
                train_loader,
                val_loader,
                mode=mode,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                save_dir=save_dir,
            )
            continue

        trainer = cfg["trainer"]

        def run_model_fn(batch, _model=model, _mode=mode):
            return trainer.run_model(_model, batch, args.device, _mode)

        run_model_fn.mode = mode

        training_loop(
            model,
            train_loader,
            val_loader,
            run_model_fn=run_model_fn,
            interval=trainer.INTERVAL,
            save_dir=save_dir,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            mode=mode,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transform",
        type=str,
        required=True,
        choices=["rotate", "shift_x", "shift_y", "shift_xy"],
    )
    parser.add_argument("--csv", type=str, default="mnist_dataset_split.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 0)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_op", "fixed_op", "learned_op"],
        choices=["no_op", "fixed_op", "learned_op"],
    )
    parser.add_argument(
        "--trained-degrees",
        type=str,
        default=None,
        help="Comma-separated list overriding default train degrees, e.g. 0,36,72",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
