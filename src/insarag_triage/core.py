"""Ordinal A–F worksite-triage classifier: data split, models, CORN loss, metrics.

Part of the insarag_triage package.

Labels: A=0 … F=5 (ordinal; A and F farthest apart).
Loss:   CORN (Shi, Cao & Raschka 2023, doi:10.1007/s10044-023-01181-9) — K-1 nested
        binary tasks "y > k", rank-consistent predictions; every sample's loss is
        weighted by w[y] = N / (K * n_y)  (inverse class prevalence, mean weight 1).
Metrics (on a split):
        confusion matrix (counts, and row-normalised = per true class),
        per-class accuracy = diagonal of the row-normalised matrix,
        overall accuracy  = sum(diagonal of row-normalised matrix) / 6  (balanced accuracy),
        MAE in grade steps: overall (micro, all images), overall (macro, mean of
        per-class MAE), and per-class MAE.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["A", "B", "C", "D", "E", "F"]
K = len(CLASSES)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# <original-photo ID>__image-<n>__ann-<m>[_jpg.rf.<hash>].<ext>; the ID may be a UUID or a number
NAME_RE = re.compile(r"^(?P<group>.+?)__image-(?P<image>\d+)__ann-(?P<ann>\d+)")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Class folder → label: "A" or e.g. "damage_level_a" / "Level-B" (final letter after a separator)
FOLDER_RE = re.compile(r"^(?:.*[_\-\s])?([A-Fa-f])$")


def folder_label(name: str) -> str | None:
    m = FOLDER_RE.match(name.strip())
    return m.group(1).upper() if m else None


# Data & split
def collect(src: Path) -> pd.DataFrame:
    """All crops under src/<any split>/<class>/<file>; Roboflow's own split is ignored."""
    rows, bad_name, bad_folder, folders = [], [], [], set()
    for p in sorted(Path(src).rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        folders.add(p.parent.name)
        label, m = folder_label(p.parent.name), NAME_RE.match(p.name)
        if label is None:
            bad_folder.append(f"{p.parent.name}/{p.name}")
        elif not m:
            bad_name.append(p.name)
        else:
            rows.append(dict(path=str(p), label=label, y=CLASSES.index(label),
                             group=m["group"].upper(), image=int(m["image"]), ann=int(m["ann"])))
    if bad_folder:
        print(f"WARNING: {len(bad_folder)} files in folders that don't end in a grade letter A–F, e.g. {bad_folder[:3]}")
    if bad_name:
        print(f"WARNING: {len(bad_name)} files whose name lacks '__image-N__ann-M', e.g. {bad_name[:3]}")
    if not rows:
        raise ValueError(f"No usable images under {src}. Class folders found: {sorted(folders)}")
    df = pd.DataFrame(rows)
    key = ["group", "image", "ann"]
    if df.groupby(key)["label"].nunique().gt(1).any():
        raise ValueError("Some crops appear with different labels — fix them in Roboflow.")
    n_dup = df.duplicated(key).sum()
    if n_dup:
        print(f"Removed {n_dup} duplicate crops.")
    return df.drop_duplicates(key).reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Grouped (by original photo) + stratified ≈71/14/14 split, fixed seed."""
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df, df["y"], df["group"])]
    df = df.copy()
    df["split"] = "train"
    df.loc[folds[0], "split"] = "test"
    df.loc[folds[1], "split"] = "val"
    assert df.groupby("group")["split"].nunique().max() == 1, "group leakage"
    return df


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(df["label"], df["split"]).reindex(index=CLASSES, columns=["train", "val", "test"], fill_value=0)
    t["total"] = t.sum(axis=1)
    t["pct"] = (100 * t["total"] / t["total"].sum()).round(1)
    return t


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    """w_c = N / (K * n_c): inverse prevalence, average weight over training images = 1."""
    counts = np.bincount(y_train, minlength=K).astype(float)
    if (counts == 0).any():
        raise ValueError(f"A class has no training images: {dict(zip(CLASSES, counts))}")
    return torch.tensor(counts.sum() / (K * counts), dtype=torch.float32)


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Pad to square (keeps the whole building, no distortion), then resize."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), tuple(int(255 * m) for m in MEAN))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


class Letterbox:
    """Picklable transform (works with DataLoader workers on every OS): RGB + letterbox to `size`."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, im: Image.Image) -> Image.Image:
        return letterbox(im.convert("RGB"), self.size)


def make_transforms(size: int, train: bool, mean=MEAN, std=STD):
    from torchvision import transforms as T
    ops = [Letterbox(size)]
    if train:
        ops += [T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),            # no vertical flip: buildings are upright
                T.RandomRotation(8),
                T.ColorJitter(brightness=0.15, contrast=0.15)]  # mild only: strong colour jitter can hide damage cues
    ops += [T.ToTensor(), T.Normalize(mean, std)]          # per-backbone normalisation (SigLIP: 0.5/0.5)
    return T.Compose(ops)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.y, self.tf = df["path"].tolist(), df["y"].to_numpy(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), int(self.y[i])


# Models
BACKBONES = {
    # timm name, kind
    "dinov2_vits14": ("vit_small_patch14_dinov2.lvd142m", "vit"),
    "dinov2_vitb14": ("vit_base_patch14_dinov2.lvd142m", "vit"),
    "convnextv2_atto": ("convnextv2_atto.fcmae_ft_in1k", "cnn"),
    # DINOv3 (Siméoni et al. 2025; DINOv3 License). Patch 16 → use sizes that are multiples of 16
    # (224 / 448 / 512). 4 register tokens + CLS; features() skips them via num_prefix_tokens.
    "dinov3_vits16": ("vit_small_patch16_dinov3.lvd1689m", "vit"),            # 21.6M
    "dinov3_vitsplus16": ("vit_small_plus_patch16_dinov3.lvd1689m", "vit"),   # 28.7M
    "dinov3_vitb16": ("vit_base_patch16_dinov3.lvd1689m", "vit"),             # ~86M
    "dinov3_convnext_tiny": ("convnext_tiny.dinov3_lvd1689m", "cnn"),         # 27.8M, distilled from the 7B teacher
    # Language-supervised (image–text) backbone: a different training paradigm for the AL committee.
    # No CLS token (attention-pool head) and mean/std 0.5 — both handled below.
    "siglip2_b16": ("vit_base_patch16_siglip_224.v2_webli", "vit"),           # 92.9M image tower
    # Supervised ImageNet-22k CNN: a different pretraining objective (class labels) AND an architecture that is
    # not distilled from DINO — replaces DINOv2-frozen in the AL committee (Gontijo-Lopes 2022; Mustafa 2020).
    "convnext_tiny_in22k": ("convnext_tiny.fb_in22k", "cnn"),                 # 27.8M, Apache-2.0
}


class TriageNet(nn.Module):
    """Backbone → pooled feature → CORN head (K-1 logits)."""

    def __init__(self, backbone: str, img_size: int = 224, pretrained: bool = True, dropout: float = 0.1,
                 head: str = "linear", hidden: int = 256):
        super().__init__()
        import timm
        name, self.kind = BACKBONES[backbone]
        kw = dict(pretrained=pretrained, num_classes=0)
        if self.kind == "vit":
            kw["img_size"] = img_size          # DINOv2 is pretrained at 518; timm resamples pos-embeddings
        self.backbone = timm.create_model(name, **kw)
        d = self.backbone.num_features
        self.feat_dim = 2 * d if self.kind == "vit" else d
        if head == "linear":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, K - 1))
        elif head == "mlp":                    # 2-layer head: room for non-linear boundaries between grades
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, hidden), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(hidden, K - 1))
        else:
            raise ValueError(f"unknown head {head!r}")

    def features(self, x):
        if self.kind == "vit":                 # CLS token ⊕ mean of patch tokens (DINOv2 linear-eval recipe)
            t = self.backbone.forward_features(x)
            n = getattr(self.backbone, "num_prefix_tokens", 1)
            if n == 0:                         # no CLS token (SigLIP): attention-pooled vector ⊕ mean of patches
                return torch.cat([self.backbone.forward_head(t, pre_logits=True), t.mean(1)], dim=1)
            return torch.cat([t[:, 0], t[:, n:].mean(1)], dim=1)
        return self.backbone(x)

    @property
    def norm(self) -> dict:
        """Normalisation the backbone was pretrained with (ImageNet for DINO/ConvNeXt, 0.5/0.5 for SigLIP)."""
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        return dict(mean=tuple(cfg.get("mean", MEAN)), std=tuple(cfg.get("std", STD)))

    def forward(self, x):
        return self.head(self.features(x))


def set_trainable(model: TriageNet, mode: str) -> None:
    """mode: 'frozen' (head only) | 'partial' (ViT: last 4 blocks; CNN: last 2 stages) | 'full'."""
    for p in model.backbone.parameters():
        p.requires_grad = mode == "full"
    if mode == "partial":
        bb = model.backbone
        parts = (list(bb.blocks[-4:]) + [bb.norm]) if model.kind == "vit" else (list(bb.stages[-2:]) + [bb.head])
        for m in parts:
            for p in m.parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


# CORN loss / prediction
def corn_loss(logits: torch.Tensor, y: torch.Tensor, w_class: torch.Tensor | None = None) -> torch.Tensor:
    """Conditional ordinal (CORN) loss with per-sample weights w_class[y].

    Task k (k = 0..K-2) asks "y > k?" and is trained only on samples with y >= k
    (conditional on having passed the previous threshold).
    """
    w = torch.ones_like(y, dtype=logits.dtype) if w_class is None else w_class.to(logits.device)[y]
    total, norm = logits.new_zeros(()), logits.new_zeros(())
    for k in range(K - 1):
        m = y >= k
        if m.any():
            target = (y[m] > k).to(logits.dtype)
            l = F.binary_cross_entropy_with_logits(logits[m, k], target, reduction="none")
            total = total + (w[m] * l).sum()
            norm = norm + w[m].sum()
    return total / norm.clamp_min(1e-8)


@torch.no_grad()
def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) = prod_{j<=k} sigmoid(logit_j); class = number of thresholds passed (> 0.5)."""
    return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)


# Metrics
def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / support[:, None]
    per_acc = np.diag(cm_row)
    err = np.abs(y_true - y_pred)
    per_mae = np.array([err[y_true == c].mean() if support[c] else np.nan for c in range(K)])
    return {
        "n": int(len(y_true)),
        "support": dict(zip(CLASSES, support.tolist())),
        "confusion_counts": cm.tolist(),
        "confusion_row_normalised": np.round(cm_row, 4).tolist(),
        "overall_accuracy_diag_over_6": float(np.nansum(per_acc) / K),
        "per_class_accuracy": dict(zip(CLASSES, np.round(per_acc, 4).tolist())),
        "plain_accuracy_all_images": float((y_true == y_pred).mean()),
        "mae_overall_micro": float(err.mean()),
        "mae_overall_macro": float(np.nanmean(per_mae)),
        "per_class_mae": dict(zip(CLASSES, np.round(per_mae, 4).tolist())),
    }


def plot_confusion(m: dict, title: str, path: Path | None = None, xlabel: str = "Predicted grade",
                   ylabel: str = "True grade", show_scores: bool = True):
    """Confusion matrix with counts and row percentages. show_scores adds balanced accuracy and MAE to the title."""
    import matplotlib.pyplot as plt
    cm, cmr = np.array(m["confusion_counts"]), np.array(m["confusion_row_normalised"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(np.nan_to_num(cmr), cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]}\n{100 * np.nan_to_num(cmr[i, j]):.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if np.nan_to_num(cmr[i, j]) > 0.5 else "black")
    ax.set_xticks(range(K), CLASSES)
    ax.set_yticks(range(K), CLASSES)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_scores:
        title = (f"{title}\nbalanced accuracy {m['overall_accuracy_diag_over_6']:.3f} · "
                 f"MAE {m['mae_overall_micro']:.3f} grades")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the row")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# Training
@dataclass
class Config:
    name: str
    backbone: str = "dinov2_vits14"
    mode: str = "frozen"          # frozen | partial | full
    img_size: int = 224
    epochs: int = 40              # frozen mode: epochs over cached features use head_epochs
    head_epochs: int = 300
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    patience: int = 8             # early stopping on val overall accuracy (diag/6); tie-break macro MAE
    head_patience: int = 40
    head: str = "linear"          # linear | mlp
    num_workers: int = 2
    extra: dict = field(default_factory=dict)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _score(m: dict) -> tuple:
    return (m["overall_accuracy_diag_over_6"], -m["mae_overall_macro"])


_FEATURE_CACHE: dict = {}   # (backbone, img_size, n_crops) → frozen features; reused across seeds/heads


@torch.no_grad()
def _extract(model, loader, dev):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(model.features(x.to(dev)).float().cpu())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


@torch.no_grad()
def _predict(model, loader, dev):
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        preds.append(corn_predict(model(x.to(dev)).float()).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def crop_id(path) -> str:
    """Stable id of a crop across exports: '<PHOTO>|<image>|<ann>' (Roboflow renames files, ids survive)."""
    m = NAME_RE.match(Path(str(path)).name)
    return f"{m['group'].upper()}|{int(m['image'])}|{int(m['ann'])}" if m else str(path)


def embed(cfg: Config, paths, pretrained: bool = True, model: TriageNet | None = None) -> torch.Tensor:
    """Frozen-backbone features for `paths`, cached PER CROP per (backbone, resolution).
    New crops are computed once; crops seen before (e.g. a pool crop that comes back labelled) are reused."""
    store = _FEATURE_CACHE.setdefault((cfg.backbone, cfg.img_size, pretrained), {})
    paths = list(map(str, paths))
    ids = [crop_id(p) for p in paths]
    todo = [(i, p) for i, p in zip(ids, paths) if i not in store]
    if todo:
        dev = device()
        model = (model or TriageNet(cfg.backbone, cfg.img_size, pretrained=pretrained)).to(dev)
        tf = make_transforms(cfg.img_size, train=False, **model.norm)
        d = pd.DataFrame({"path": [p for _, p in todo], "y": 0})
        loader = torch.utils.data.DataLoader(CropDataset(d, tf), batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, pin_memory=dev.type == "cuda")
        feats, _ = _extract(model, loader, dev)
        store.update({i: f for (i, _), f in zip(todo, feats)})
    return torch.stack([store[i] for i in ids]) if ids else torch.empty(0)


def save_feature_cache(path: Path):
    torch.save({k: v for k, v in _FEATURE_CACHE.items()}, path)


def load_feature_cache(path: Path):
    if Path(path).exists():
        for k, v in torch.load(path).items():
            _FEATURE_CACHE.setdefault(k, {}).update(v)


def frozen_features(cfg: Config, df: pd.DataFrame, pretrained: bool = True, model: TriageNet | None = None):
    """(ftr, ytr, fva, yva, fte, yte) for a frozen backbone (per-crop cache, see embed())."""
    out = []
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        out += [embed(cfg, d["path"], pretrained, model), torch.as_tensor(d["y"].to_numpy().copy(), dtype=torch.long)]
    return tuple(out)


def train_head(head: nn.Module, ftr, ytr, fva, yva, w, cfg: Config, seed: int, dev) -> list:
    """Train a head on cached features; keep the epoch with the best validation score (in place)."""
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    best, best_state, bad, history = None, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(cfg.head_epochs):
        head.train()
        for idx in torch.randperm(len(ytr), generator=g).split(256):
            loss = corn_loss(head(ftr[idx].to(dev)), ytr[idx].to(dev), w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            mv = evaluate(yva.numpy(), corn_predict(head(fva.to(dev))).cpu().numpy())
        history.append(dict(epoch=ep, loss=loss.item(), val_acc=mv["overall_accuracy_diag_over_6"],
                            val_mae_macro=mv["mae_overall_macro"]))
        if best is None or _score(mv) > _score(best):
            best, best_state, bad = mv, {k: v.detach().clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.head_patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return history


def run(cfg: Config, df: pd.DataFrame, seed: int, out_dir: Path, pretrained: bool = True) -> dict:
    """Train one config/seed; select the epoch by validation; evaluate val + test once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    dev = device()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    w = class_weights(tr["y"].to_numpy())
    model = TriageNet(cfg.backbone, cfg.img_size, pretrained=pretrained, head=cfg.head).to(dev)
    if cfg.extra.get("init_from"):            # warm start: continue from earlier weights (e.g. previous AL round)
        model.load_state_dict(torch.load(cfg.extra["init_from"], map_location="cpu"))
        if cfg.extra.get("reset_head"):       # keep the adapted backbone, re-learn the head (Nikishin 2022; Kumar 2022)
            for m in model.head.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
    set_trainable(model, cfg.mode)
    eval_tf = make_transforms(cfg.img_size, train=False, **model.norm)
    mk = lambda d, tf, sh: torch.utils.data.DataLoader(CropDataset(d, tf), batch_size=cfg.batch_size, shuffle=sh,
                                                       num_workers=cfg.num_workers, pin_memory=dev.type == "cuda")
    t0, history = time.time(), []

    if cfg.mode == "frozen":
        # Backbone is fixed → compute features once (no augmentation), train only the head.
        # Features depend only on backbone + resolution, so they are cached across seeds and heads.
        ftr, ytr, fva, yva, fte, yte = frozen_features(cfg, df, pretrained, model=model)
        head = model.head.to(dev)
        history = train_head(head, ftr, ytr, fva, yva, w, cfg, seed, dev)
        with torch.no_grad():
            m_val = evaluate(yva.numpy(), corn_predict(head(fva.to(dev))).cpu().numpy())
            y_te, p_te = yte.numpy(), corn_predict(head(fte.to(dev))).cpu().numpy()
    else:
        train_loader = mk(tr, make_transforms(cfg.img_size, train=True, **model.norm), True)
        val_loader, test_loader = mk(va, eval_tf, False), mk(te, eval_tf, False)
        groups = [{"params": [p for p in model.head.parameters()], "lr": cfg.lr_head},
                  {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.lr_backbone}]
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        steps, warm = cfg.epochs * len(train_loader), cfg.warmup_epochs * len(train_loader)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        best, best_state, bad = None, None, 0
        for ep in range(cfg.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits = model(x)
                loss = corn_loss(logits.float(), y, w)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            yv, pv = _predict(model, val_loader, dev)
            mv = evaluate(yv, pv)
            history.append(dict(epoch=ep, loss=loss.item(), val_acc=mv["overall_accuracy_diag_over_6"],
                                val_mae_macro=mv["mae_overall_macro"]))
            print(f"  ep {ep:02d} loss {loss.item():.3f} val acc(diag/6) {mv['overall_accuracy_diag_over_6']:.3f} "
                  f"MAE macro {mv['mae_overall_macro']:.3f}")
            if best is None or _score(mv) > _score(best):
                best, best_state, bad = mv, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        m_val = evaluate(*_predict(model, val_loader, dev))
        y_te, p_te = _predict(model, test_loader, dev)

    m_test = evaluate(y_te, p_te)
    if cfg.mode == "frozen":   # backbone = unchanged pretrained weights; store only the trained head
        torch.save(model.head.state_dict(), out_dir / "head.pt")
    else:
        torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame({"path": te["path"].values, "y_true": y_te, "y_pred": p_te}).to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    result = dict(config=asdict(cfg), seed=seed, minutes=round((time.time() - t0) / 60, 2),
                  class_weights=dict(zip(CLASSES, np.round(w.numpy(), 3).tolist())), val=m_val, test=m_test)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    plot_confusion(m_test, f"{cfg.name} seed {seed} — TEST", out_dir / "confusion_test.png")
    return result
