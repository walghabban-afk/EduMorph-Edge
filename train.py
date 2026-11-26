"""
train.py - Train the MSC-Trans engagement classifier (Appendix A.3).

AdamW (lr 2e-5, weight decay 0.01), batch size 32, 50 epochs, cosine schedule
with linear warm-up over the first 5 epochs. Model selection uses macro-averaged
F1 on a validation subset of the training students; the held-out test students
are used only by evaluate.py.

  python train.py                       # synthetic surrogate (pipeline check)
  python train.py --data labelled.npz   # anonymized labelling set
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from config import (CHECKPOINT_DIR, MSC_BATCH_SIZE, MSC_EPOCHS, MSC_LR,
                    MSC_WARMUP_EPOCHS, MSC_WEIGHT_DECAY, RANDOM_SEED)
from dataset import get_dataloaders
from model import MSCTrans
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help=".npz labelling set (default: synthetic)")
    p.add_argument("--epochs", type=int, default=MSC_EPOCHS)
    p.add_argument("--batch_size", type=int, default=MSC_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=MSC_LR)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true",
                   help="use deterministic algorithms (set CUBLAS_WORKSPACE_CONFIG=:4096:8 on GPU)")
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y, yhat = [], []
    for face, click, text, label in loader:
        logits = model(face.to(device), click.to(device), text.to(device))
        y.extend(label.tolist())
        yhat.extend(logits.argmax(1).cpu().tolist())
    return y, yhat


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    loaders, split = get_dataloaders(args.data, args.batch_size, args.seed)

    model = MSCTrans().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=MSC_WEIGHT_DECAY)
    steps_per_epoch = len(loaders["train"])
    total, warm = args.epochs * steps_per_epoch, MSC_WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for face, click, text, label in loaders["train"]:
            opt.zero_grad()
            loss = criterion(model(face.to(device), click.to(device), text.to(device)), label.to(device))
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(label)
        y, yhat = predict(model, loaders["val"], device)
        val_f1 = f1_score(y, yhat, average="macro")
        history.append({"epoch": epoch, "train_loss": running / len(loaders["train"].dataset),
                        "val_macro_f1": val_f1})
        print(f"epoch {epoch:3d}  train loss {history[-1]['train_loss']:.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best:
            best = val_f1
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1,
                        "seed": args.seed, "data": args.data or "synthetic"},
                       os.path.join(CHECKPOINT_DIR, "best_msc_trans.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), "w") as f:
        json.dump({"seed": args.seed, "data": args.data or "synthetic",
                   "n_students": {k: int(len(v)) for k, v in split.items()},
                   "history": history}, f, indent=2)
    print(f"best validation macro-F1 {best:.4f}; checkpoint in {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
