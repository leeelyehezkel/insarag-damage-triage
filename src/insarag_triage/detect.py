"""Find buildings in a photo with SAM 3 text prompts, cut each one out, and evaluate prompt settings.

Pipeline (settings in configs/detect.yaml):
    every prompt is run once at a low base confidence -> per-prompt score thresholds -> class-agnostic NMS across
    prompts -> size filter relative to the photo -> crop with padding -> quality filter for grading.

SAM 3 is loaded through Hugging Face transformers (>= 5). The checkpoint `facebook/sam3` is gated: accept its terms
on Hugging Face and log in. In the study SAM 3 ran as a Roboflow Workflow block; scores from the two deployments
may differ slightly, so re-check the thresholds visually on a few photos (notebook 00).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from . import pool as P

COLS = ["photo", "prompt", "score", "x", "y", "w", "h", "img_w", "img_h"]


def load_config(path: str | Path = "configs/detect.yaml") -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def load_sam3(model_id: str = "facebook/sam3", device: str | None = None):
    from transformers import Sam3Model, Sam3Processor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, Sam3Processor.from_pretrained(model_id)


def open_photo(path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


# Running SAM 3

@torch.no_grad()
def detect(model, processor, image: Image.Image, prompts, base_confidence: float = 0.2,
           mask_threshold: float = 0.5, photo: str = "") -> pd.DataFrame:
    """Boxes (x, y, w, h in pixels) and scores for every prompt at the base confidence.
    The image is encoded once and reused for every prompt."""
    img_in = processor(images=image, return_tensors="pt").to(model.device)
    vision = model.get_vision_features(pixel_values=img_in.pixel_values.to(model.dtype))
    sizes = img_in.get("original_sizes").tolist()
    rows = []
    for prompt in prompts:
        txt = processor(text=prompt, return_tensors="pt").to(model.device)
        out = model(vision_embeds=vision, **txt)
        res = processor.post_process_instance_segmentation(out, threshold=base_confidence,
                                                           mask_threshold=mask_threshold, target_sizes=sizes)[0]
        for (x1, y1, x2, y2), s in zip(res["boxes"].tolist(), res["scores"].tolist()):
            rows.append((photo, prompt, float(s), x1, y1, x2 - x1, y2 - y1, image.width, image.height))
    return pd.DataFrame(rows, columns=COLS)


def detect_many(model, processor, photos, prompts, base_confidence: float = 0.2) -> pd.DataFrame:
    """Run every prompt on every photo once. Higher thresholds and prompt combinations can then be compared
    offline, without calling SAM 3 again."""
    out = [detect(model, processor, open_photo(p), prompts, base_confidence, photo=Path(p).name) for p in photos]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)


# Selecting boxes

def select(dets: pd.DataFrame, thresholds: dict, nms_iou: float = 0.4, size: dict | None = None) -> pd.DataFrame:
    """Per-prompt thresholds, then class-agnostic NMS across prompts (per photo), then the relative size filter."""
    import torchvision
    d = dets[dets.prompt.isin(thresholds)]
    d = d[d.score >= d.prompt.map(thresholds)]
    keep = []
    for _, g in d.groupby("photo"):
        b = torch.tensor(g[["x", "y", "w", "h"]].to_numpy(), dtype=torch.float32)
        b[:, 2:] += b[:, :2]
        k = torchvision.ops.nms(b, torch.tensor(g.score.to_numpy(), dtype=torch.float32), nms_iou)
        keep.extend(g.index[k.numpy()])
    d = d.loc[keep]
    if size:
        ok = ((d.w / d.img_w >= size["min_width"]) & (d.h / d.img_h >= size["min_height"])
              & (d.w * d.h / (d.img_w * d.img_h) >= size["min_area"]))
        d = d[ok]
    return d.sort_values(["photo", "score"], ascending=[True, False]).reset_index(drop=True)


def crop_photo(path, model, processor, cfg: dict, out_dir, aerial=None) -> pd.DataFrame:
    """Detect, select, crop and quality-check the buildings of one photo. `aerial` is an optional function
    photo -> bool; aerial photos return no crops. Crops are written as <photo>__image-0__ann-<k>.jpg."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = open_photo(path)
    if aerial is not None and aerial(img):
        return pd.DataFrame()
    dets = detect(model, processor, img, list(cfg["prompts"]), cfg["base_confidence"], photo=path.name)
    boxes = select(dets, cfg["prompts"], cfg["nms_iou"], cfg["size_filter"])
    stem = path.stem.replace("__", "_")
    rows = []
    for k, r in boxes.iterrows():
        crop = P.crop_with_padding(img, (r.x, r.y, r.w, r.h), cfg["padding"])
        f = out_dir / f"{stem}__image-0__ann-{k}.jpg"
        crop.save(f, quality=95)
        rows.append(dict(photo=path.name, path=str(f), prompt=r.prompt, score=r.score,
                         x=r.x, y=r.y, w=r.w, h=r.h, width=crop.width, height=crop.height))
    crops = pd.DataFrame(rows)
    if crops.empty:
        return crops
    q = cfg["quality"]
    crops = P.quality_table(crops)
    crops["quality_ok"] = P.quality_ok(crops, q["max_aspect"], q["min_side"], q["lum_lo"], q["lum_hi"], q["min_std"]).to_numpy()
    return crops


# Scoring against hand-drawn boxes

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xywh boxes a (n, 4) and b (m, 4)."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(ax1[:, None], bx1), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(ay1[:, None], by1), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.5) -> pd.DataFrame:
    """One-to-one matching per photo (Hungarian, on IoU), class-agnostic: any predicted box can match any
    reference box. Returns the reference boxes with a `found` column, plus the number of unmatched predictions."""
    from scipy.optimize import linear_sum_assignment
    gt = gt.copy(); gt["found"] = False
    fp = 0
    for photo, g in gt.groupby("photo"):
        p = pred[pred.photo == photo]
        if p.empty:
            continue
        m = _iou(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= iou
        gt.loc[g.index[r[ok]], "found"] = True
        fp += len(p) - int(ok.sum())
    fp += len(pred[~pred.photo.isin(gt.photo.unique())])
    gt.attrs["fp"] = fp
    return gt


def score(pred: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.5, grade_col: str = "grade") -> dict:
    """Precision, recall, F2 (recall weighted 4× as much as precision) and recall per grade."""
    m = match(pred, gt, iou)
    tp, fn, fp = int(m.found.sum()), int((~m.found).sum()), m.attrs["fp"]
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f2 = 5 * prec * rec / (4 * prec + rec) if prec and rec else 0.0
    out = dict(precision=prec, recall=rec, f2=f2, tp=tp, fp=fp, fn=fn, fp_per_photo=fp / max(1, gt.photo.nunique()))
    if grade_col in m:
        out.update({f"recall_{g}": v for g, v in m.groupby(grade_col)["found"].mean().items()})
    return out


def region_scores(pred: pd.DataFrame, gt: pd.DataFrame, found_at: float = 0.5, grid: int = 512) -> dict:
    """Coverage measures that don't require one box per building: share of reference boxes at least `found_at`
    covered by the union of predicted boxes, share of the predicted area inside reference boxes, and share of
    reference area covered. Computed on a grid of about `grid` pixels per side."""
    found, inside, pred_area, cov_area, gt_area = [], 0, 0, 0, 0
    for photo, g in gt.groupby("photo"):
        W, H = int(g.img_w.iloc[0]), int(g.img_h.iloc[0])
        s = grid / max(W, H)
        def mask(df):
            mk = np.zeros((int(H * s) + 1, int(W * s) + 1), bool)
            for _, r in df.iterrows():
                mk[int(r.y * s):int((r.y + r.h) * s) + 1, int(r.x * s):int((r.x + r.w) * s) + 1] = True
            return mk
        pm, gm = mask(pred[pred.photo == photo]), mask(g)
        for _, r in g.iterrows():
            box = np.zeros_like(gm); box[int(r.y * s):int((r.y + r.h) * s) + 1, int(r.x * s):int((r.x + r.w) * s) + 1] = True
            found.append((pm & box).sum() / max(1, box.sum()) >= found_at)
        inside += (pm & gm).sum(); pred_area += pm.sum(); cov_area += (pm & gm).sum(); gt_area += gm.sum()
    return dict(found=float(np.mean(found)) if found else np.nan,
                area_precision=float(inside / pred_area) if pred_area else np.nan,
                area_recall=float(cov_area / gt_area) if gt_area else np.nan)


def search_combinations(dets: pd.DataFrame, gt: pd.DataFrame, prompts, threshold: float, max_size: int = 3,
                        nms_iou: float = 0.4) -> pd.DataFrame:
    """Score every combination of up to `max_size` prompts at one common threshold (offline, from cached boxes)."""
    rows = []
    for k in range(1, max_size + 1):
        for combo in combinations(prompts, k):
            s = score(select(dets, {p: threshold for p in combo}, nms_iou), gt)
            rows.append(dict(prompts=" + ".join(combo), size=k, **{x: s[x] for x in ("precision", "recall", "f2", "tp", "fp", "fn")}))
    return pd.DataFrame(rows).sort_values("f2", ascending=False).reset_index(drop=True)


def threshold_table(dets: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Boxes per photo for each prompt and threshold (before NMS)."""
    n = max(1, dets["photo"].nunique())
    out = {t: dets[dets.score >= t].groupby("prompt").size() / n for t in thresholds}
    return pd.DataFrame(out).fillna(0).round(2).rename_axis(index="prompt", columns="threshold")


# Looking at the boxes

def show_detections(photo_path, boxes: pd.DataFrame, gt: pd.DataFrame | None = None, ax=None, title=None):
    """Draw one photo with its predicted boxes (coloured by prompt, with score) and, optionally, the reference
    boxes as dashed white outlines."""
    import matplotlib.pyplot as plt
    name = Path(photo_path).name
    ax = ax or plt.gca()
    ax.imshow(open_photo(photo_path)); ax.axis("off")
    names = sorted(boxes.prompt.unique()) if len(boxes) else []
    cmap = {p: plt.cm.tab10(i % 10) for i, p in enumerate(names)}
    if gt is not None:
        for _, r in gt[gt.photo == name].iterrows():
            ax.add_patch(plt.Rectangle((r.x, r.y), r.w, r.h, fill=False, lw=1.5, ls="--", color="white"))
    for _, r in boxes[boxes.photo == name].iterrows():
        ax.add_patch(plt.Rectangle((r.x, r.y), r.w, r.h, fill=False, lw=2, color=cmap[r.prompt]))
        ax.text(r.x, r.y, f"{r.score:.2f}", color="white", fontsize=7, va="bottom",
                bbox=dict(facecolor=cmap[r.prompt], alpha=0.75, pad=1, lw=0))
    handles = [plt.Line2D([], [], color=cmap[p], lw=3, label=p) for p in names]
    if gt is not None:
        handles.append(plt.Line2D([], [], color="white", lw=1.5, ls="--", label="reference box"))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.8)
    ax.set_title(title or f"photo {name[:24]}", fontsize=9)
    return ax
