"""Resolve train/validation manifest paths: prefer flat ``data_root/*.npy``, else legacy ``data_root/0/*.npy``."""
from __future__ import annotations

import os


def resolve_train_npy_path(data_root: str) -> str:
    for rel in ("train.npy", os.path.join("0", "train.npy")):
        p = os.path.join(data_root, rel)
        if os.path.isfile(p):
            return p
    return os.path.join(data_root, "train.npy")


def resolve_validation_npy_path(data_root: str) -> str:
    for name in ("validation.npy", "test.npy"):
        flat = os.path.join(data_root, name)
        if os.path.isfile(flat):
            return flat
        sub = os.path.join(data_root, "0", name)
        if os.path.isfile(sub):
            return sub
    return os.path.join(data_root, "validation.npy")


def resolve_best_model_pt(model_save_root: str) -> str:
    """``Best_model.pt`` in save root, or legacy ``0/Best_model.pt``."""
    flat = os.path.join(model_save_root, "Best_model.pt")
    if os.path.isfile(flat):
        return flat
    legacy = os.path.join(model_save_root, "0", "Best_model.pt")
    if os.path.isfile(legacy):
        return legacy
    return flat
