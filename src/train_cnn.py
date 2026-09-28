"""
train_cnn.py - fine-tune a pre-trained ResNet-18 on the handwriting images.

    python -m src.train_cnn                  (GPU: ~3 min | laptop CPU: ~20-40 min)
    python -m src.train_cnn --epochs 5       (quick test run)

No GPU? Use Google Colab (free GPU) - see README, "Training on Colab".

How training works (one EPOCH = one pass over all 3,120 training images):
  for each small batch of 32 images:
      1. forward pass   -> the model predicts probabilities for 78 classes
      2. loss           -> cross-entropy: how wrong were the predictions?
      3. backward pass  -> compute how each weight contributed to the error
      4. optimiser step -> nudge every weight slightly to reduce the error
  after each epoch: measure accuracy on the VALIDATION set.
  We keep the epoch with the best validation accuracy (not the last one),
  and stop early if it has not improved for CNN_PATIENCE epochs.
  -> this is how we fight OVERFITTING (memorising the training images).

Other anti-overfitting tricks used here:
  - Data augmentation: small random rotations, shifts, slants and pen
    thickness changes, so the model sees "new" handwriting every epoch.
    (NO horizontal flips - a flipped word is not a valid word!)
  - Label smoothing (0.1): the model is trained to be 90% sure instead of
    100% sure, which makes its confidence scores more realistic.
  - Weight decay (AdamW): keeps weights small = simpler model.
"""
import argparse
import random
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config
from src.cnn_model import build_model, get_device, image_to_tensor, normalise
from src.data_loader import get_class_names, load_all
from src.preprocessing import load_gray, preprocess

# Geometric augmentation, applied on the (1, H, W) tensor. fill=1.0 = white.
GEOMETRIC_AUGMENT = transforms.RandomAffine(
    degrees=5,               # rotate up to +-5 degrees
    translate=(0.05, 0.05),  # shift up to 5%
    scale=(0.9, 1.05),       # zoom slightly
    shear=10,                # slant, like italic handwriting
    fill=1.0,
)


def random_pen_thickness(img, rng):
    """ECE-style augmentation with morphology (ink is dark, paper is white):
    erosion (min filter) grows the dark ink = thicker pen,
    dilation (max filter) shrinks it = thinner pen."""
    r = rng.random()
    kernel = np.ones((2, 2), np.uint8)
    if r < 0.15:
        return cv2.erode(img, kernel)
    if r < 0.30:
        return cv2.dilate(img, kernel)
    return img


class HandwritingDataset(Dataset):
    """Holds preprocessed images in memory and serves (tensor, label) pairs."""

    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            img = random_pen_thickness(img, self.rng)
        x = image_to_tensor(img)
        if self.augment:
            x = GEOMETRIC_AUGMENT(x)
        return normalise(x), int(self.labels[i])


def set_seed(seed):
    """Same seed -> same random choices -> reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                 # 1. forward pass
            loss = loss_fn(logits, y)         # 2. loss
            if training:
                optimizer.zero_grad()
                loss.backward()               # 3. backward pass
                optimizer.step()              # 4. update weights
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss (lower is better)")
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="validation")
    axes[1].set_title("Accuracy (higher is better)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "cnn_training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CNN_LEARNING_RATE)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from random weights (to show why transfer learning helps)")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    train, val, _ = load_all()
    classes = get_class_names(train)
    class_to_id = {c: i for i, c in enumerate(classes)}
    print("Preprocessing images...")
    x_train = [preprocess(load_gray(p), config.CNN_IMG_SIZE) for p in train["image_path"]]
    x_val = [preprocess(load_gray(p), config.CNN_IMG_SIZE) for p in val["image_path"]]
    y_train = train["medicine"].map(class_to_id).values
    y_val = val["medicine"].map(class_to_id).values

    train_loader = DataLoader(HandwritingDataset(x_train, y_train, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(HandwritingDataset(x_val, y_val, augment=False),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- model ------------------------------------------------------------
    try:
        model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    except Exception as e:  # usually: no internet to download the weights
        raise SystemExit(f"Could not download pre-trained weights ({e}).\n"
                         "Check your internet connection, or run with --no-pretrained.")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine schedule: learning rate starts at lr and smoothly decays to ~0,
    # big steps early (learn fast), tiny steps late (fine adjustments).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- training loop ----------------------------------------------------
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch = -1.0, 0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        for key, value in zip(history, [epoch, tr_loss, tr_acc, va_loss, va_acc]):
            history[key].append(value)
        marker = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": config.CNN_IMG_SIZE,
                "arch": "resnet18",
                "pretrained": not args.no_pretrained,
                "val_accuracy": va_acc,
                "epoch": epoch,
            }, config.CNN_MODEL_PATH)
            marker = "  <- best so far, saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}  ({time.time() - start:.0f}s){marker}")
        if epoch - best_epoch >= config.CNN_PATIENCE:
            print(f"No improvement for {config.CNN_PATIENCE} epochs -> early stopping.")
            break

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.REPORTS_DIR / "cnn_history.csv", index=False)
    plot_history(history)
    print(f"\nBest validation accuracy {best_acc:.3f} at epoch {best_epoch}.")
    print(f"Model saved to {config.CNN_MODEL_PATH.relative_to(config.ROOT)}")
    print("Training curves: reports/figures/cnn_training_curves.png")


if __name__ == "__main__":
    main()
