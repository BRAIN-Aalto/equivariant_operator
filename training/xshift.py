from .base import *
import torch

# x-shift uses the generic utilities in base.py with this cycle/interval setup.
CYCLE = 14
INTERVAL = 2


def run_model(model, batch, device, mode):
    """Compute x-shift training loss for one batch."""
    img_a, deg_a, cls, img_t, deg_t = [b.to(device) for b in batch]

    idx_a = deg_a // INTERVAL
    idx_t = deg_t // INTERVAL

    I = torch.eye(model.block_size, device=device)

    if mode == "no_op":
        emb_a, pred_a = model(img_a)
        emb_t = model.encode(img_t)
        loss = criterion1(emb_a, emb_t) + criterion2(pred_a, cls)

    elif mode == "fixed_op":
        emb_a, pred_a = model(img_a, skip_transform=False, degs=-idx_a)
        emb_t = model.encode(img_t, skip_transform=False, degs=-idx_t)
        loss = criterion1(emb_a, emb_t) + criterion2(pred_a, cls)

    elif mode == "learned_op":
        emb_a, pred_a = model(img_a, skip_transform=False, degs=-idx_a)
        emb_t = model.encode(img_t, skip_transform=False, degs=-idx_t)
        loss = criterion1(emb_a, emb_t) + criterion2(pred_a, cls)
        pk = torch.linalg.matrix_power(model.P, CYCLE)
        loss += criterion1(I, pk)

    else:
        raise ValueError(f"Unknown mode {mode}")

    return loss, img_a.size(0)
