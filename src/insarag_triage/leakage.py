"""Find the same picture in two sets (e.g. training and test) even when the file names differ.

A photo can exist twice in a source dataset under two names (X and X__1_), and an active-learning loop can pick a
crop that is also in a test set. A name-based photo split misses both. The fingerprint is a 16×16 average hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def fingerprint(path) -> str:
    im = Image.open(path)
    im.draft("L", (64, 64))                       # fast reduced JPEG decoding
    g = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32)
    return hashlib.md5((g > g.mean()).tobytes()).hexdigest()


def find_leaks(sets: dict[str, list], eval_sets: list[str], train_sets: list[str]) -> pd.DataFrame:
    """sets: name → list of image paths. Returns one row per image whose fingerprint appears in both an evaluation
    set and a training set (role = 'eval' or 'train')."""
    rows = [(name, str(p), Path(p).name, fingerprint(p)) for name, paths in sets.items() for p in paths]
    fp = pd.DataFrame(rows, columns=["set", "path", "file", "fp"])
    shared = set(fp.loc[fp.set.isin(eval_sets), "fp"]) & set(fp.loc[fp.set.isin(train_sets), "fp"])
    out = fp[fp.fp.isin(shared)].copy()
    out["role"] = np.where(out.set.isin(eval_sets), "eval", "train")
    return out.sort_values(["fp", "role"]).reset_index(drop=True)
