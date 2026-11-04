"""Post-hoc analysis of finished runs (no retraining):

1. bootstrap_ci   — 95 % confidence intervals for every test metric, resampling whole
                    ORIGINAL PHOTOS with replacement (crops of one photo are not independent).
2. error_gallery  — contact sheet + CSV of misclassified test crops, grouped by
                    (true → predicted) pair, using the consensus prediction over seeds.

Adds "within-1" accuracy (prediction exact or one grade off), per class and balanced.
Depends only on triage_lib (CLASSES, K, NAME_RE, evaluate, letterbox).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import core as L


def extended_metrics(y_true, y_pred) -> dict:
    """triage_lib.evaluate + within-1-grade accuracy (per class, balanced = mean over classes)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = L.evaluate(y_true, y_pred)
    ok1 = np.abs(y_true - y_pred) <= 1
    per = np.array([ok1[y_true == c].mean() if (y_true == c).any() else np.nan for c in range(L.K)])
    m["per_class_within1"] = dict(zip(L.CLASSES, np.round(per, 4).tolist()))
    m["within1_balanced"] = float(np.nanmean(per))
    m["within1_all_images"] = float(ok1.mean())
    return m


def exif_orientation_report(paths) -> pd.Series:
    """Count images by EXIF orientation tag (1 = normal; 3 = 180°; 6/8 = 90°; missing = no tag).
    Our loader does not apply this tag, so any value other than 1/missing means the model saw the
    image rotated relative to how a photo viewer shows it."""
    from PIL import Image
    vals = []
    for p in paths:
        try:
            vals.append(Image.open(p).getexif().get(0x0112, "missing"))
        except Exception:
            vals.append("unreadable")
    return pd.Series(vals, name="exif_orientation").astype(str).value_counts()


def report(results: list, run_root: Path, cfg_name: str, seeds, out_dir: Path, n_boot: int = 2000):
    """Per-class table, pooled test confusion matrix and bootstrap CIs for one config."""
    try:
        from IPython.display import display
    except ImportError:
        display = print
    best = [r for r in results if r["config"]["name"] == cfg_name]
    rows = []
    for c in L.CLASSES:
        acc = [r["test"]["per_class_accuracy"][c] for r in best]
        mae = [r["test"]["per_class_mae"][c] for r in best]
        rows.append(dict(cls=c, support=best[0]["test"]["support"][c],
                         accuracy=f"{np.nanmean(acc):.3f} ± {np.nanstd(acc):.3f}",
                         MAE=f"{np.nanmean(mae):.3f} ± {np.nanstd(mae):.3f}"))
    display(pd.DataFrame(rows).set_index("cls"))
    pred = load_test_predictions(Path(run_root) / cfg_name, seeds)
    pooled = L.evaluate(pred["y_true"], pred["y_pred"])
    L.plot_confusion(pooled, f"{cfg_name}: TEST, pooled over seeds {list(seeds)}",
                     Path(out_dir) / f"confusion_test_pooled_{cfg_name}.png")
    ci = bootstrap_ci(pred, n_boot=n_boot)
    ci.to_csv(Path(out_dir) / f"bootstrap_ci_test_{cfg_name}.csv")
    display(ci)
    return pred, ci


def learning_curve(df: pd.DataFrame, backbone: str = "dinov2_vits14", img_size: int = 224,
                   fractions=(0.25, 0.5, 0.75, 1.0), repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Validation performance of the frozen-backbone head vs. amount of training data.

    For each fraction, a random subset of training PHOTOS (groups) is drawn (all crops of a photo
    together), the head is trained exactly as in the real runs (class weights from the subset,
    early stopping on validation), and VALIDATION metrics are recorded. The test set is not used.
    """
    import torch
    cfg = L.Config(f"lc_{backbone}_{img_size}", backbone=backbone, mode="frozen", img_size=img_size)
    ftr, ytr, fva, yva, _, _ = L.frozen_features(cfg, df)
    dev = L.device()
    tr = df[df.split == "train"].reset_index(drop=True)          # same row order as ftr/ytr
    groups = tr["group"].unique()
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        for rep in range(repeats):
            for _ in range(100):                                    # redraw until every class is present
                chosen = rng.choice(groups, size=max(1, round(frac * len(groups))), replace=False)
                idx = np.flatnonzero(tr["group"].isin(chosen).to_numpy())
                if np.bincount(ytr[idx].numpy(), minlength=L.K).min() > 0:
                    break
            else:
                raise RuntimeError(f"could not draw a subset with all classes at fraction {frac}")
            L.seed_all(seed + rep)
            # same linear CORN head as TriageNet(head="linear"); the backbone is not needed (features cached)
            head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(ftr.shape[1], L.K - 1)).to(dev)
            w = L.class_weights(ytr[idx].numpy())
            L.train_head(head, ftr[idx], ytr[idx], fva, yva, w, cfg, seed + rep, dev)
            with torch.no_grad():
                m = L.evaluate(yva.numpy(), L.corn_predict(head(fva.to(dev))).cpu().numpy())
            rows.append(dict(img_size=img_size, fraction=frac, repeat=rep, n_train_crops=len(idx),
                             n_train_photos=len(chosen), val_acc=m["overall_accuracy_diag_over_6"],
                             val_mae_macro=m["mae_overall_macro"],
                             **{f"acc_{c}": v for c, v in m["per_class_accuracy"].items()}))
    return pd.DataFrame(rows)


def plot_learning_curve(lc: pd.DataFrame, path: Path | None = None):
    """Left: overall validation accuracy (mean ± std over repeats). Right: per-class accuracy."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for size, d in lc.groupby("img_size"):
        g = d.groupby("fraction").agg(n=("n_train_crops", "mean"), m=("val_acc", "mean"), s=("val_acc", "std"))
        a1.errorbar(g["n"], g["m"], yerr=g["s"], marker="o", capsize=4, label=f"{size} px")
    a1.set(xlabel="training crops", ylabel="validation accuracy (diag/6)",
           title="Overall (mean ± std over repeats)")
    a1.grid(alpha=0.3)
    a1.legend()
    biggest = lc["img_size"].max()
    d = lc[lc["img_size"] == biggest]
    g = d.groupby("fraction")[["n_train_crops"] + [f"acc_{c}" for c in L.CLASSES]].mean()
    for c in L.CLASSES:
        a2.plot(g["n_train_crops"], g[f"acc_{c}"], marker="o", label=c)
    a2.set(xlabel="training crops", ylabel="validation accuracy", title=f"Per class ({biggest} px, mean over repeats)")
    a2.grid(alpha=0.3)
    a2.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


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


def consensus(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per test crop: true label, each seed's prediction, consensus = median over seeds."""
    wide = pred.pivot_table(index=["file", "path", "group", "y_true"], columns="seed", values="y_pred").reset_index()
    seed_cols = [c for c in wide.columns if isinstance(c, (int, np.integer))]
    wide["y_cons"] = np.round(wide[seed_cols].median(axis=1)).astype(int)
    wide["seed_preds"] = wide[seed_cols].astype(int).apply(lambda r: ",".join(L.CLASSES[v] for v in r), axis=1)
    wide["true"] = wide["y_true"].map(lambda v: L.CLASSES[int(v)])
    wide["pred"] = wide["y_cons"].map(lambda v: L.CLASSES[int(v)])
    wide["abs_err"] = (wide["y_cons"] - wide["y_true"]).abs()
    return wide.drop(columns=seed_cols)


def error_gallery(pred: pd.DataFrame, out_png: Path, out_csv: Path, max_pairs: int = 8, per_pair: int = 8,
                  thumb: int = 160):
    """Contact sheet of consensus errors: one row per (true→pred) pair, most frequent pairs first."""
    import matplotlib.pyplot as plt
    from PIL import Image

    c = consensus(pred)
    err = c[c["y_cons"] != c["y_true"]].copy()
    err["pair"] = err["true"] + "→" + err["pred"]
    counts = err["pair"].value_counts()
    err = err.assign(pair_count=err["pair"].map(counts)).sort_values(["pair_count", "pair", "abs_err"],
                                                                     ascending=[False, True, False])
    err[["file", "group", "true", "pred", "seed_preds", "abs_err", "pair", "pair_count", "path"]].to_csv(out_csv, index=False)

    pairs = counts.index[:max_pairs]
    if len(pairs) == 0:
        print("No consensus errors.")
        return counts
    fig, axes = plt.subplots(len(pairs), per_pair, figsize=(1.9 * per_pair, 2.25 * len(pairs)), squeeze=False)
    for r, pair in enumerate(pairs):
        rows = err[err["pair"] == pair].head(per_pair)
        for k in range(per_pair):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(rows):
                row = rows.iloc[k]
                try:
                    ax.imshow(L.letterbox(Image.open(row["path"]), thumb))
                except FileNotFoundError:
                    ax.text(0.5, 0.5, "image not\non disk", ha="center", va="center", fontsize=7)
                ax.set_title(f"{row['file'][:18]}…\nseeds {row['seed_preds']}", fontsize=6)
        axes[r, 0].text(-0.15, 0.5, f"{pair}\n(n={counts[pair]})", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    fig.suptitle("Misclassified test crops (consensus over seeds) — true→predicted", fontsize=11)
    fig.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    return counts
