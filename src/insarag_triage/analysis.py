"""Post-hoc analysis of finished runs (no retraining):

1. bootstrap_ci   — 95 % confidence intervals for every test metric, resampling whole
                    ORIGINAL PHOTOS with replacement (crops of one photo are not independent).
2. error_gallery  — contact sheet + CSV of misclassified test crops, grouped by
                    (true → predicted) pair, using the consensus prediction over seeds.

Adds "within-1" accuracy (prediction exact or one grade off), per class and balanced.
Builds on core (CLASSES, K, NAME_RE, evaluate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import core as L


def extended_metrics(y_true, y_pred) -> dict:
    """core.evaluate + within-1-grade accuracy (per class, balanced = mean over classes)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = L.evaluate(y_true, y_pred)
    ok1 = np.abs(y_true - y_pred) <= 1
    per = np.array([ok1[y_true == c].mean() if (y_true == c).any() else np.nan for c in range(L.K)])
    m["per_class_within1"] = dict(zip(L.CLASSES, np.round(per, 4).tolist()))
    m["within1_balanced"] = float(np.nanmean(per))
    m["within1_all_images"] = float(ok1.mean())
    return m


def load_test_predictions(run_dir: Path, seeds) -> pd.DataFrame:
    """Long table: one row per (test crop, seed) with group = original-photo ID."""
    frames = []
    for s in seeds:
        p = pd.read_csv(Path(run_dir) / f"seed{s}" / "test_predictions.csv")
        p["seed"] = s
        frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    df["file"] = df["path"].map(lambda x: Path(x).name)
    df["group"] = df["file"].map(lambda f: L.NAME_RE.match(f)["group"].upper())
    return df


SCALARS = ["overall_accuracy_diag_over_6", "mae_overall_micro", "mae_overall_macro",
           "within1_balanced", "plain_accuracy_all_images"]
PER_CLASS = ["per_class_accuracy", "per_class_mae", "per_class_within1"]


def _flat(m: dict) -> dict:
    out = {k: m[k] for k in SCALARS}
    for k in PER_CLASS:
        out.update({f"{k}:{c}": v for c, v in m[k].items()})
    # Overall accuracy recomputed as mean over classes PRESENT (identical to diag/6 when all 6 are
    # present; avoids a downward bias in the rare resample that happens to miss a class entirely).
    out["overall_accuracy_diag_over_6"] = float(np.nanmean(list(m["per_class_accuracy"].values())))
    return out


def bootstrap_ci(pred: pd.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Cluster bootstrap over original photos; all seeds' predictions of a photo travel together."""
    rng = np.random.default_rng(seed)
    groups = pred["group"].unique()
    by_group = {g: idx for g, idx in pred.groupby("group").indices.items()}
    yt, yp = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    point = _flat(extended_metrics(yt, yp))
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        samples.append(_flat(extended_metrics(yt[idx], yp[idx])))
    s = pd.DataFrame(samples)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return pd.DataFrame({
        "estimate": pd.Series(point),
        f"ci{int(level * 100)}_low": s.apply(lambda c: np.nanpercentile(c, lo)),
        f"ci{int(level * 100)}_high": s.apply(lambda c: np.nanpercentile(c, hi)),
    }).round(3)
