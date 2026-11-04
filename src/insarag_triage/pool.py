"""Build the UNLABELLED crop pool for active learning from the box-annotated Roboflow dataset.

Roboflow project `damage-region-dataset` (object detection, class `damage_region`):
  1. download a version in COCO format (use a version WITHOUT resize, so crops keep full resolution);
  2. crop every damage_region box with padding (fraction of box width/height on each side, clipped);
  3. name crops exactly like the labelled set:  <original-photo ID>__image-<image id>__ann-<ann id>.jpg
     so grouping by original photo keeps working;
  4. EXCLUDE every original photo already present in the labelled dataset (manifest.csv), so no crop of a
     val/test photo can ever enter training (leakage) and no already-labelled photo is labelled twice;
  5. write the crops + pool_manifest.csv (to the working directory).

Uploading to a Roboflow CLASSIFICATION project (labelling UI only — no Roboflow training) is a separate
step: upload_to_roboflow(), with a batch name and optional tags (e.g. the model's predicted grade).
Credentials are never stored here; read the key from the ROBOFLOW_API_KEY environment variable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

# Roboflow export names look like  <original>_jpg.rf.<hash>.jpg ; recover <original>.
RF_SUFFIX = re.compile(r"_(?:jpe?g|png|heic|webp|bmp)\.rf\.[0-9a-f]+(?:\.(?:jpe?g|png|webp|bmp))?$", re.I)


def original_id(file_name: str) -> str:
    """'216567436941332_jpg.rf.ab12.jpg' → '216567436941332'; 'IMG_1.jpg' → 'IMG_1' (upper-cased)."""
    name = Path(file_name).name
    stripped = RF_SUFFIX.sub("", name)
    return (stripped if stripped != name else Path(name).stem).upper()


def download_detection(api_key: str, workspace: str, project: str, version: int, location: Path) -> Path:
    from roboflow import Roboflow
    location = Path(location)
    if not location.exists():
        Roboflow(api_key=api_key).workspace(workspace).project(project).version(version).download(
            "coco", location=str(location))
    return location


def iter_coco(root: Path, category: str = "damage_region"):
    """Yield (image_path, image_id, original_id, [x, y, w, h], ann_id) over all export splits."""
    for ann_file in sorted(Path(root).rglob("_annotations.coco.json")):
        coco = json.loads(ann_file.read_text())
        cats = {c["id"] for c in coco["categories"] if c["name"] == category}
        if not cats:
            raise ValueError(f"No category {category!r} in {ann_file}; found {[c['name'] for c in coco['categories']]}")
        imgs = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            if a["category_id"] in cats:
                im = imgs[a["image_id"]]
                orig = (im.get("extra") or {}).get("name") or im["file_name"]
                yield ann_file.parent / im["file_name"], f"{ann_file.parent.name}-{im['id']}", original_id(orig), a["bbox"], a["id"]


def crop_with_padding(img: Image.Image, bbox, pad: float) -> Image.Image:
    x, y, w, h = bbox
    px, py = pad * w, pad * h
    W, H = img.size
    box = (max(0, round(x - px)), max(0, round(y - py)), min(W, round(x + w + px)), min(H, round(y + h + py)))
    return img.crop(box)


def estimate_padding(coco_root: Path, labelled: pd.DataFrame, category: str = "damage_region",
                     max_checks: int = 300) -> pd.DataFrame:
    """Estimate the padding used for the LABELLED crops by matching them to their boxes.

    A labelled crop  <photo>__image-N__ann-M.jpg  is matched to the box with annotation id M in the same
    photo (falling back to the box whose aspect ratio is closest). Padding per side ≈ (crop − box) / 2 / box,
    computed separately for width and height. Boxes cut off at the image edge are skipped (padding clipped).
    """
    boxes = {}
    for path, _, group, bbox, ann_id in iter_coco(coco_root, category):
        boxes.setdefault(group, []).append((ann_id, bbox, path))
    rows = []
    for _, r in labelled.head(max_checks * 5).iterrows():
        if r["group"] not in boxes or len(rows) >= max_checks:
            continue
        cands = boxes[r["group"]]
        cw, ch = Image.open(r["path"]).size
        exact = [c for c in cands if c[0] == r["ann"]]
        ann_id, (x, y, w, h), img_path = exact[0] if exact else min(
            cands, key=lambda c: abs(c[1][2] / c[1][3] - cw / ch))
        W, H = Image.open(img_path).size
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                                  # touching the edge → padding would be clipped
        rows.append(dict(file=Path(r["path"]).name, matched="ann-id" if exact else "aspect",
                         box_w=w, box_h=h, crop_w=cw, crop_h=ch,
                         pad_w=(cw / w - 1) / 2, pad_h=(ch / h - 1) / 2))
    est = pd.DataFrame(rows)
    if est.empty:
        print("No labelled photo found in the detection dataset — cannot estimate padding.")
    else:
        print(f"{len(est)} labelled crops matched ({(est.matched == 'ann-id').mean():.0%} by annotation id). "
              f"Median padding per side: width {est.pad_w.median():.1%}, height {est.pad_h.median():.1%} "
              f"(IQR width {est.pad_w.quantile(.25):.1%}–{est.pad_w.quantile(.75):.1%}).")
        if est.pad_w.median() < -0.02 or est.pad_h.median() < -0.02:
            print("Crops are SMALLER than their boxes → the labelled crops were probably resized; padding unknown.")
    return est


def build_pool(coco_root: Path, out_dir: Path, labelled_manifest: Path | None, pad: float = 0.10,
               min_side: int = 32, category: str = "damage_region") -> pd.DataFrame:
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    known = set()
    if labelled_manifest and Path(labelled_manifest).exists():
        known = set(pd.read_csv(labelled_manifest)["group"].astype(str).str.upper())
    rows, skipped_known, skipped_small = [], 0, 0
    seen = set()
    for path, img_key, group, bbox, ann_id in iter_coco(coco_root, category):
        if group in known:
            skipped_known += 1
            continue
        key = (group, tuple(round(v) for v in bbox))
        if key in seen:                      # same photo + box exported in two splits
            continue
        seen.add(key)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        crop = crop_with_padding(img, bbox, pad)
        if min(crop.size) < min_side:
            skipped_small += 1
            continue
        img_no = int(str(img_key).split("-")[-1])          # COCO image id within its export split
        name = f"{group}__image-{img_no}__ann-{ann_id}.jpg"
        crop.save(out_dir / "crops" / name, quality=95)
        rows.append(dict(file=name, path=str(out_dir / "crops" / name), group=group, image_key=img_key,
                         ann_id=ann_id, bbox=json.dumps([round(v, 1) for v in bbox]), pad=pad,
                         width=crop.size[0], height=crop.size[1]))
    pool = pd.DataFrame(rows)
    pool.to_csv(out_dir / "pool_manifest.csv", index=False)
    print(f"pool: {len(pool)} crops from {pool['group'].nunique() if len(pool) else 0} photos | "
          f"skipped {skipped_known} boxes from photos already in the labelled set, {skipped_small} tiny crops")
    return pool


def upload_to_roboflow(paths, api_key: str, workspace: str, project: str, batch_name: str,
                       tags_per_image: list[list[str]] | None = None):
    """Upload crops (unlabelled) to an existing Roboflow classification project for labelling in the UI."""
    from roboflow import Roboflow
    proj = Roboflow(api_key=api_key).workspace(workspace).project(project)
    ok, failed = 0, []
    for i, p in enumerate(paths):
        try:
            proj.upload(image_path=str(p), batch_name=batch_name,
                        tag_names=(tags_per_image[i] if tags_per_image else []), num_retry_uploads=2)
            ok += 1
        except Exception as e:                 # keep going; report at the end
            failed.append((str(p), str(e)[:120]))
    print(f"uploaded {ok}/{len(paths)} to {workspace}/{project} as batch {batch_name!r}")
    if failed:
        print(f"{len(failed)} failed, e.g. {failed[:2]}")
    return failed


def near_duplicates(pool: pd.DataFrame, labelled: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """For every pool crop: the most similar LABELLED crop (cosine similarity of frozen DINOv3 features).

    Used when photo names differ between datasets, to detect the same building/photo appearing in both
    (a leakage risk if the labelled match is in val/test). Returns one row per pool crop, sorted by similarity.
    """
    import numpy as np
    import torch
    from . import core as L
    cfg = cfg or L.Config("dup_check", backbone="dinov3_vits16", mode="frozen", img_size=224)
    a = torch.nn.functional.normalize(L.embed(cfg, pool["path"]).float(), dim=1)
    b = torch.nn.functional.normalize(L.embed(cfg, labelled["path"]).float(), dim=1)
    sim = a @ b.T
    best, j = sim.max(dim=1)
    j = j.numpy()
    out = pool[["file", "path", "group"]].copy()
    out["max_sim"] = best.numpy()
    out["match_file"] = labelled["path"].map(lambda p: Path(p).name).to_numpy()[j]
    out["match_path"] = labelled["path"].to_numpy()[j]
    out["match_split"] = labelled["split"].to_numpy()[j]
    out["match_group"] = labelled["group"].to_numpy()[j]
    return out.sort_values("max_sim", ascending=False).reset_index(drop=True)


def show_pairs(dups: pd.DataFrame, n: int = 24, path=None):
    """Side-by-side pool crop | best labelled match, highest similarity first."""
    import matplotlib.pyplot as plt
    from . import core as L
    rows = dups.head(n)
    fig, axs = plt.subplots(len(rows), 2, figsize=(5, 2.4 * len(rows)), squeeze=False)
    for (ax1, ax2), (_, r) in zip(axs, rows.iterrows()):
        ax1.imshow(L.letterbox(Image.open(r.path), 160)); ax1.set_title(f"POOL {r.file[:22]}", fontsize=6)
        ax2.imshow(L.letterbox(Image.open(r.match_path), 160))
        ax2.set_title(f"{r.match_split.upper()} sim={r.max_sim:.3f}", fontsize=7)
        ax1.axis("off"); ax2.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90)
    return fig


def _iou_contain(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0, inter / max(1e-9, min(aw * ah, bw * bh))


def dedupe_boxes(pool: pd.DataFrame, iou: float = 0.5, contain: float = 0.8) -> pd.DataFrame:
    """Within each source image, drop a box that overlaps a LARGER kept box (IoU ≥ iou, or ≥ `contain` of the
    smaller box inside the larger one). Keeps the larger box (more of the building)."""
    keep = []
    for _, g in pool.groupby("image_key"):
        boxes = g.assign(_b=g["bbox"].map(json.loads)).assign(_a=lambda d: d["_b"].map(lambda b: b[2] * b[3]))
        kept = []
        for i, r in boxes.sort_values("_a", ascending=False).iterrows():
            ov = [_iou_contain(r["_b"], k) for k in kept]
            if not any(i_ >= iou or c_ >= contain for i_, c_ in ov):
                kept.append(r["_b"]); keep.append(i)
    out = pool.loc[sorted(keep)].reset_index(drop=True)
    print(f"dedupe_boxes: {len(pool)} → {len(out)} crops ({len(pool) - len(out)} overlapping boxes removed)")
    return out


def quality_table(pool: pd.DataFrame) -> pd.DataFrame:
    """Per crop: aspect ratio, short side, mean / std luminance (0–1) from a small thumbnail."""
    import numpy as np
    lum_m, lum_s = [], []
    for p in pool["path"]:
        g = np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        lum_m.append(float(g.mean())); lum_s.append(float(g.std()))
    w, h = pool["width"].astype(float), pool["height"].astype(float)
    return pool.assign(aspect=np.maximum(w, h) / np.minimum(w, h), short_side=np.minimum(w, h),
                       lum_mean=lum_m, lum_std=lum_s)


def quality_ok(q: pd.DataFrame, max_aspect: float = 3.0, min_side: int = 96, lum_lo: float = 0.12,
               lum_hi: float = 0.90, min_std: float = 0.05) -> pd.Series:
    ok = ((q["aspect"] <= max_aspect) & (q["short_side"] >= min_side) &
          q["lum_mean"].between(lum_lo, lum_hi) & (q["lum_std"] >= min_std))
    print(f"quality filter: {int(ok.sum())}/{len(q)} usable | removed: aspect>{max_aspect}: {int((q.aspect > max_aspect).sum())}, "
          f"short side<{min_side}: {int((q.short_side < min_side).sum())}, too dark/bright/flat: "
          f"{int((~q.lum_mean.between(lum_lo, lum_hi) | (q.lum_std < min_std)).sum())}")
    return ok
