# Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges

Codebase for experiments on latent equivariant operators for MNIST classification under geometric transformations.

## Core Idea
The model learns latent representations that are either:
- `no_op`: no transformation operator (baseline)
- `fixed_op`: fixed cyclic latent operator
- `learned_op`: learned latent operator with periodicity regularization

Supported transforms:
- `rotate`
- `shift_x`
- `shift_y`
- `shift_xy`

## Project Layout
```text
.
├── datasets/
│   ├── mnist_download.py      # download + export raw MNIST PNGs
│   ├── mnist_gen.py           # generate transformed RGB images
│   ├── make_split_csv.py      # build train/val/test CSV split
│   ├── polygon_dataset.py     # main training dataset
│   └── compound_dataset.py
├── models/
│   └── linear_classifier.py
├── training/
│   ├── base.py
│   ├── rotate.py
│   ├── xshift.py
│   ├── yshift.py
│   └── xyshift.py
├── mnist_train_cls.py         # main training entrypoint
├── knn_xshift.py              # x-shift evaluation script
├── knn_yshift.py              # y-shift evaluation script
└── README.md
```

## Installation
```bash
pip install torch torchvision tqdm numpy pandas pillow
```

## Data Pipeline
Run from repository root.

### 1) Download raw MNIST
```bash
python datasets/mnist_download.py
```
Creates:
- `mnist_local/train/<digit>/*.png`
- `mnist_local/test/<digit>/*.png`

### 2) Generate transformed images
```bash
python datasets/mnist_gen.py
```
Creates:
- `mnist_combined/train/<digit>/*.png`
- `mnist_combined/test/<digit>/*.png`

### 3) Build split CSV
```bash
python datasets/make_split_csv.py
```
Creates:
- `mnist_dataset_split.csv`

Expected CSV columns:
- `file_path` (relative to `mnist_local`, e.g. `train/3/123.png`)
- `class` (integer)
- `split` (`train`, `val`, `test`)

## Training
Main command:
```bash
python mnist_train_cls.py \
  --transform {rotate|shift_x|shift_y|shift_xy} \
  [--modes no_op fixed_op learned_op] \
  [--epochs 20] [--lr 1e-3] [--batch-size 512] \
  [--device cuda|cpu] [--num-workers N] [--prefetch-factor 8] \
  [--checkpoint-root checkpoints] \
  [--csv mnist_dataset_split.csv] \
  [--trained-degrees 0,36,72]
```

Examples:
```bash
python mnist_train_cls.py --transform rotate
python mnist_train_cls.py --transform shift_x --modes fixed_op
python mnist_train_cls.py --transform shift_y --epochs 40 --lr 5e-4 --batch-size 256
```

Useful args:
- `--csv` path to split file (default: `mnist_dataset_split.csv`)
- `--epochs` (default: `20`)
- `--lr` (default: `1e-3`)
- `--batch-size` (default: `512`)
- `--num-workers` (default: `os.cpu_count()`)
- `--prefetch-factor` (default: `8`)
- `--no-pin-memory`
- `--checkpoint-root` (default: `checkpoints`)
- `--modes` subset of `no_op fixed_op learned_op`
- `--trained-degrees` comma list override (example: `0,36,72`)
- `--device` (default: auto `cuda` if available, else `cpu`)

Show full CLI:
```bash
python mnist_train_cls.py --help
```

## Outputs
- Checkpoints are saved under `checkpoints/` (or `--checkpoint-root`).
- Naming follows `mnist_<transform>_cls_<mode>/best.pth` and `latest.pth`.

## Evaluation
Run after training checkpoints exist:
```bash
python knn_xshift.py
python knn_yshift.py
```

## Notes
- Current `PolygonDataset` filters to classes `< 9` (digits `0-8`).
- `shift_xy` uses the transform-specific loop in `training/xyshift.py`.

## Citation
```bibtex
@inproceedings{
dinh2026latent,
title={Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges},
author={Minh T. Dinh and Stephane Deny},
booktitle={ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling},
year={2026},
url={https://openreview.net/forum?id=81gVwLVcXQ}
}
```
