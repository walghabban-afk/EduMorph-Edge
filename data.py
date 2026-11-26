"""
synthetic_data.py - Synthetic surrogate for the engagement-classifier labelling set.

The deployment data involve minors and are not distributed with this package
(see data/README.md). This module generates a synthetic dataset with the same
structure as the labelling set described in Section III-A of the paper
(240 students x 60 five-minute windows, three engagement classes, three input
streams), so that the training and evaluation code can be run end to end.

The synthetic data are NOT a model of the real data: class frequencies,
feature distributions and separability are arbitrary choices made here so
that the pipeline has a learnable signal. Scores obtained on synthetic data
are therefore not comparable with the results reported in the paper.

Requires only numpy.
"""

import numpy as np

from config import (
    LABEL_SET_STUDENTS, WINDOWS_PER_STUDENT, TRAIN_FRACTION, VAL_FRACTION_OF_TRAIN,
    FACE_AU_DIM, CLICK_DIM, TEXT_DIM, MSC_N_CLASSES, RANDOM_SEED,
)

# Arbitrary class frequencies for the synthetic surrogate (0=Disengaged,
# 1=Transitioning, 2=Engaged); not estimates from the study.
SYNTHETIC_CLASS_PRIOR = (0.25, 0.35, 0.40)


def generate(n_students=LABEL_SET_STUDENTS, windows=WINDOWS_PER_STUDENT,
             seed=RANDOM_SEED, text_dim=TEXT_DIM, signal=0.6):
    """Return a dict of numpy arrays: student_id, label, face, click, text."""
    rng = np.random.default_rng(seed)
    n = n_students * windows
    student_id = np.repeat(np.arange(n_students), windows)
    label = rng.choice(MSC_N_CLASSES, size=n, p=SYNTHETIC_CLASS_PRIOR)
    # class centroids and a per-student offset (students differ systematically)
    cent_face = rng.normal(0, 1, (MSC_N_CLASSES, FACE_AU_DIM))
    cent_click = rng.normal(0, 1, (MSC_N_CLASSES, CLICK_DIM))
    cent_text = rng.normal(0, 1, (MSC_N_CLASSES, text_dim))
    stu_face = rng.normal(0, 0.5, (n_students, FACE_AU_DIM))
    stu_click = rng.normal(0, 0.5, (n_students, CLICK_DIM))
    face = (signal * cent_face[label] + stu_face[student_id]
            + rng.normal(0, 1, (n, FACE_AU_DIM))).astype(np.float32)
    click = (signal * cent_click[label] + stu_click[student_id]
             + rng.normal(0, 1, (n, CLICK_DIM))).astype(np.float32)
    text = (0.5 * signal * cent_text[label]
            + rng.normal(0, 1, (n, text_dim))).astype(np.float32)
    return {"student_id": student_id, "label": label.astype(np.int64),
            "face": face, "click": click, "text": text}


def student_split(n_students=LABEL_SET_STUDENTS, seed=RANDOM_SEED,
                  train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION_OF_TRAIN):
    """Student-level split: 68% training students (a validation subset is drawn
    from these for model selection) and 32% held-out test students."""
    rng = np.random.default_rng(seed + 1)
    ids = rng.permutation(n_students)
    n_train = int(round(train_fraction * n_students))
    train_all, test = ids[:n_train], ids[n_train:]
    n_val = int(round(val_fraction * n_train))
    val, train = train_all[:n_val], train_all[n_val:]
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


if __name__ == "__main__":
    data = generate()
    split = student_split()
    print(f"windows: {len(data['label'])}; students: {LABEL_SET_STUDENTS}")
    for k, v in split.items():
        print(f"  {k:5s}: {len(v)} students")
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])
