import os
import torch
from tqdm import tqdm

from .base import (
    IMG_SIZE,
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    criterion1,
    criterion2,
    save_checkpoint,
)

CYCLE = 14
INTERVAL = 2


def evaluate_model(model, loader, device, mode="no_op"):
    """Evaluate xy-shift model where x and y operators are staged."""

    model.eval()

    total_loss = 0
    total_samples = 0
    total_correct = 0

    with torch.no_grad():

        for batch in loader:

            img_a, deg_a, cls, img_t, deg_t = [b.to(device) for b in batch]

            idx_a = deg_a // INTERVAL
            idx_t = deg_t // INTERVAL

            if mode == "no_op":

                _, logits1 = model(img_a)
                _, logits2 = model(img_t)

            else:

                _, logits1 = model(img_a, degs=[-idx_a, None])
                _, logits2 = model(img_t, degs=[None, -idx_t])

            loss = criterion2(logits1, cls) + criterion2(logits2, cls)

            total_loss += loss.item() * img_a.size(0)
            total_samples += img_a.size(0) * 2

            preds1 = logits1.argmax(dim=1)
            preds2 = logits2.argmax(dim=1)

            total_correct += (preds1 == cls).sum().item()
            total_correct += (preds2 == cls).sum().item()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    print(f"[{mode.upper()}] Eval: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    return avg_loss, acc


def run_model(model, batch, device, mode, I):
    """Compute xy-shift alignment/classification loss for one batch."""

    img_a, deg_a, cls, img_t, deg_t = [b.to(device) for b in batch]

    # Degree values are stored as pixel offsets; normalize to operator index.
    idx_a = deg_a // INTERVAL
    idx_t = deg_t // INTERVAL

    if mode == "no_op":

        emb_a = model.encode(img_a)
        emb_t = model.encode(img_t)

    else:

        emb_a = model.encode(img_a, degs=[-idx_a, None])
        emb_t = model.encode(img_t, degs=[None, -idx_t])

    pred_a = model.decode(emb_a)
    pred_t = model.decode(emb_t)

    loss = (
        criterion1(emb_a, emb_t)
        + criterion2(pred_a, cls)
        + criterion2(pred_t, cls)
    )

    if mode == "learned_op":

        pk_loss = 0
        for P in model.P:
            pk = torch.linalg.matrix_power(P, CYCLE)
            pk_loss += criterion1(I, pk)

        loss += pk_loss

    return loss, img_a.size(0)


def train_one_epoch(model, loader, optimizer, device, mode):
    """Train one epoch for xy-shift setting."""

    model.train()

    epoch_loss = 0
    total_samples = 0

    I = None if mode == "no_op" else torch.eye(model.block_size, device=device)

    for batch in tqdm(loader, desc=f"Training [{mode}]", leave=False):

        loss, batch_size = run_model(model, batch, device, mode, I)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

    return epoch_loss / total_samples


def training_loop(
    model,
    train_loader,
    val_loader,
    mode="no_op",
    device="cuda",
    epochs=EPOCHS,
    lr=0.001,
    save_dir=None,
):
    """XY-shift specific loop with its own checkpoint path convention."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    if save_dir is None:
        save_dir = f"checkpoints/mnist_xyshift_cls_{mode}"

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, device, mode)

        val_loss, _ = evaluate_model(model, val_loader, device, mode)

        is_best = val_loss < best_val

        if is_best:
            best_val = val_loss

        checkpoint_names = ["latest.pth", "best.pth"] if is_best else ["latest.pth"]

        for name in checkpoint_names:

            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(save_dir, name),
                train_loss,
                val_loss,
            )

        print(
            f"Epoch {epoch+1} | Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} {'[NEW BEST]' if is_best else ''}"
        )

    best_ckpt = torch.load(os.path.join(save_dir, "best.pth"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model
