"""
dataset.py - PyTorch datasets for the MSC-Trans engagement classifier.

Two sources are supported:
  * the synthetic surrogate from synthetic_data.py (default), and
  * a user-supplied .npz file with the same arrays (student_id, label, face,
    click, text), e.g. the anonymized labelling set obtained from the authors
    under the data-access conditions in data/README.md.
Students are split 68% / 32% (training / held-out test) at the student level;
a validation subset for model selection is drawn from the training students.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import synthetic_data
from config import RANDOM_SEED


class EngagementWindows(Dataset):
    def __init__(self, arrays, student_ids):
        mask = np.isin(arrays["student_id"], student_ids)
        self.face = torch.from_numpy(arrays["face"][mask])
        self.click = torch.from_numpy(arrays["click"][mask])
        self.text = torch.from_numpy(arrays["text"][mask])
        self.label = torch.from_numpy(arrays["label"][mask])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.face[i], self.click[i], self.text[i], self.label[i]


def load_arrays(npz_path=None, seed=RANDOM_SEED):
    if npz_path:
        with np.load(npz_path) as z:
            arrays = {k: z[k] for k in ("student_id", "label", "face", "click", "text")}
        # remap arbitrary student identifiers to contiguous 0..N-1
        _, arrays["student_id"] = np.unique(arrays["student_id"], return_inverse=True)
        return arrays
    return synthetic_data.generate(seed=seed)


def get_dataloaders(npz_path=None, batch_size=32, seed=RANDOM_SEED):
    arrays = load_arrays(npz_path, seed)
    n_students = int(arrays["student_id"].max()) + 1
    split = synthetic_data.student_split(n_students=n_students, seed=seed)
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(EngagementWindows(arrays, split["train"]), batch_size=batch_size,
                            shuffle=True, generator=g),
        "val": DataLoader(EngagementWindows(arrays, split["val"]), batch_size=batch_size),
        "test": DataLoader(EngagementWindows(arrays, split["test"]), batch_size=batch_size),
    }
    return loaders, split
