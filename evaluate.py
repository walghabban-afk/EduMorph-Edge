"""
evaluate.py - Evaluate a trained MSC-Trans checkpoint on the held-out test
students (student-level split, Section III-A).

Reports per-class precision, recall and F1 and the macro-averaged F1, the
metric reported in Table A.II. On the synthetic surrogate the scores only
confirm that the pipeline works; they are not comparable with Table A.II.
The study's statistical results are verified separately by verify_statistics.py.

  python evaluate.py [--data labelled.npz] [--checkpoint checkpoints/best_msc_trans.pt]
"""

import argparse
import json
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import CHECKPOINT_DIR, ENGAGE_LABELS, RANDOM_SEED, RESULTS_DIR
from dataset import get_dataloaders
from model import MSCTrans
from train import predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_msc_trans.pt"))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint {args.checkpoint} not found; run train.py first.")
    device = torch.device(args.device)
    loaders, split = get_dataloaders(args.data, 64, args.seed)
    model = MSCTrans().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])
    y, yhat = predict(model, loaders["test"], device)

    names = [ENGAGE_LABELS[i] for i in range(3)]
    labels = [0, 1, 2]
    report = classification_report(y, yhat, labels=labels, target_names=names, digits=3,
                                   output_dict=True, zero_division=0)
    print(classification_report(y, yhat, labels=labels, target_names=names, digits=3, zero_division=0))
    out = {"data": args.data or "synthetic", "test_students": int(len(split["test"])),
           "test_windows": len(y), "macro_f1": f1_score(y, yhat, labels=labels, average="macro", zero_division=0),
           "per_class": {k: report[k] for k in names},
           "confusion_matrix": confusion_matrix(y, yhat, labels=labels).tolist()}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "synthetic" if args.data is None else os.path.splitext(os.path.basename(args.data))[0]
    path = os.path.join(RESULTS_DIR, f"classifier_evaluation_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"macro-F1 = {out['macro_f1']:.3f}; saved to {path}")


if __name__ == "__main__":
    main()
