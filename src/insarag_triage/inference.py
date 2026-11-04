"""Load a released model (local folder or Hugging Face repo) and grade building crops A–F.

A release folder contains:
    config.json          backbone, input size, head type, grades, list of weight files
    model_seed*.pt       one state dict per training seed (predictions are averaged over seeds)
    LICENSE-DINOv3.md    the DINOv3 License (required with DINOv3-derived weights)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import core as L


@dataclass
class Released:
    config: dict
    models: list            # one TriageNet per seed, in eval mode
    device: torch.device


def _resolve(source: str | Path) -> Path:
    p = Path(source)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download        # only needed for Hub repos
    return Path(snapshot_download(repo_id=str(source)))


def load_model(source: str | Path, device: str | torch.device | None = None) -> Released:
    """source = a local release folder, or a Hugging Face repo id such as 'user/insarag-triage-dinov3-vits16'."""
    root = _resolve(source)
    cfg = json.loads((root / "config.json").read_text())
    dev = torch.device(device) if device else L.device()
    models = []
    for f in cfg["weights"]:
        m = L.TriageNet(cfg["backbone"], cfg["img_size"], pretrained=False, head=cfg.get("head", "linear"))
        m.load_state_dict(torch.load(root / f, map_location="cpu"))
        models.append(m.to(dev).eval())
    return Released(cfg, models, dev)


@torch.no_grad()
def predict(rel: Released, paths, batch_size: int = 16, num_workers: int = 2) -> pd.DataFrame:
    """Grade every image in `paths`. Returns one row per image:
    grade (A–F), expected_grade (0–5, a continuous severity), P(y > t) for each threshold, and P(grade)."""
    paths = [str(p) for p in paths]
    cfg, K = rel.config, len(L.CLASSES)
    tf = L.make_transforms(cfg["img_size"], train=False, **rel.models[0].norm)
    ds = L.CropDataset(pd.DataFrame({"path": paths, "y": 0}), tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    cum_seeds = []
    for m in rel.models:
        outs = [torch.sigmoid(m(x.to(rel.device)).float()).cpu() for x, _ in loader]
        cum_seeds.append(torch.cumprod(torch.cat(outs), dim=1).numpy())     # P(y > t), rank-consistent (CORN)
    cum = np.mean(cum_seeds, axis=0)                                         # average over seeds
    grade = (cum > 0.5).sum(1)
    ext = np.concatenate([np.ones((len(cum), 1)), cum, np.zeros((len(cum), 1))], axis=1)
    probs = ext[:, :-1] - ext[:, 1:]                                         # P(y = k) = P(y > k-1) − P(y > k)
    out = pd.DataFrame({"path": paths, "grade": [L.CLASSES[g] for g in grade], "expected_grade": cum.sum(1)})
    for i in range(K - 1):
        out[f"P(>{L.CLASSES[i]})"] = cum[:, i]
    for k in range(K):
        out[f"P({L.CLASSES[k]})"] = probs[:, k]
    return out
