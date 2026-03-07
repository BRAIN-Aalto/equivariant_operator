import torch
import torch.nn.functional as F

from .base import (
    criterion1,
    criterion2,
    criterion2_noreduction,
)

CYCLE = 14
INTERVAL = 2


def run_model(model, batch, device, mode):
    """Compute y-shift training loss for one batch."""

    img_a, deg_a, cls, img_t, deg_t = [b.to(device) for b in batch]

    # Convert pixel shifts to operator indices (shift//2 for this setup).
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

    elif mode == "fixed_op_no_deg":

        emb_a, pred_a = model(img_a, skip_transform=False, degs=None)
        emb_t = model.encode(img_t).unsqueeze(1)

        mse_per_k = torch.mean((emb_a - emb_t) ** 2, dim=-1)

        temp = 0.1
        weights = torch.softmax(-mse_per_k / temp, dim=1)

        B = emb_a.shape[0]
        best_k = mse_per_k.argmin(dim=1)
        batch_idx = torch.arange(B, device=emb_a.device)

        pred_a = pred_a.view(B, CYCLE, -1)[batch_idx, best_k]

        loss = (
            criterion1(emb_a[batch_idx, best_k], emb_t.squeeze(1))
            + criterion2(pred_a, cls)
        )

    else:
        raise ValueError(f"Unknown mode {mode}")

    return loss, img_a.size(0)
