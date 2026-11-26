"""
inference.py - One pass of the adaptive loop for a single five-minute window:
classify engagement with MSC-Trans, then choose an intervention with the
debiased LinUCB bandit (Section III-A, Appendix A).

  python inference.py [--checkpoint checkpoints/best_msc_trans.pt]
Inputs are drawn from the synthetic surrogate unless a real window is supplied
programmatically via run_window().
"""

import argparse
import json
import os

import numpy as np
import torch

import synthetic_data
from config import CHECKPOINT_DIR, ENGAGE_LABELS, FACE_AU_DIM, CLICK_DIM, RANDOM_SEED, RESULTS_DIR
from model import DebiasedLinUCB, MSCTrans


def run_window(model, bandit, face, click, text, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(face)[None].to(device),
                       torch.as_tensor(click)[None].to(device),
                       torch.as_tensor(text)[None].to(device))
        probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()
    state = int(probs.argmax())
    context = np.concatenate([face, click, probs]).astype(np.float32)
    arm = bandit.select_arm(context)
    return state, probs, arm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_msc_trans.pt"))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    model = MSCTrans()
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
        source = args.checkpoint
    else:
        source = "untrained weights (run train.py to create a checkpoint)"
    bandit = DebiasedLinUCB(in_dim=FACE_AU_DIM + CLICK_DIM + 3)

    data = synthetic_data.generate(n_students=1, windows=1, seed=args.seed)
    state, probs, arm = run_window(model, bandit, data["face"][0], data["click"][0], data["text"][0])
    out = {"model": source, "engagement_state": ENGAGE_LABELS[state],
           "probabilities": {ENGAGE_LABELS[i]: float(probs[i]) for i in range(3)},
           "intervention": bandit.intervention_name(arm)}
    print(json.dumps(out, indent=2))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "inference_example.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
