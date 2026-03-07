import os
import torch
import torch.nn as nn
from tqdm import tqdm

IMG_SIZE = 28
INPUT_DIM = 3 * IMG_SIZE * IMG_SIZE
LATENT_DIM = 70

criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
criterion2_noreduction = nn.CrossEntropyLoss(reduction="none")

EPOCHS = 20


def save_checkpoint(model, optimizer, epoch, path, train_loss, val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )

def evaluate_model(model, loader, device, mode, interval):
    """Evaluate classification loss/accuracy for one dataloader."""

    model.eval()

    total_loss = 0
    total_samples = 0
    total_correct = 0

    with torch.no_grad():

        for batch in loader:

            imgs, degs, labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

            if mode == "no_op":

                _, logits = model(imgs)

            else:
                # Convert raw degree (e.g. 72) into operator index (e.g. 2).
                idx_anchor = degs // interval
                _, logits = model(imgs, skip_transform=False, degs=-idx_anchor)

            loss = criterion2(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    print(f"[{mode.upper()}] Eval: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device, run_model_fn):
    """
    Train one epoch using a task-specific callback.
    run_model_fn(batch) must return (loss, batch_size).
    """

    model.train()

    epoch_loss = 0
    total_samples = 0

    for batch in tqdm(loader, leave=False):

        loss, batch_size = run_model_fn(batch)

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
    run_model_fn,
    interval,
    save_dir,
    device="cuda",
    epochs=EPOCHS,
    lr=0.001,
    mode=None,
):
    """Generic training loop with latest/best checkpoint tracking."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eval_mode = mode if mode is not None else getattr(run_model_fn, "mode")

    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, run_model_fn
        )

        val_loss, _ = evaluate_model(
            model,
            val_loader,
            device,
            eval_mode,
            interval,
        )

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
            f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
            f"{'[NEW BEST]' if is_best else ''}"
        )

    best_ckpt = torch.load(os.path.join(save_dir, "best.pth"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model
