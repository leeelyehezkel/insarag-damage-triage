"""Build the UNLABELLED crop pool for active learning from the box-annotated Roboflow dataset.

Roboflow project `damage-region-dataset` (object detection, class `damage_region`):
  1. download a version in COCO format (use a version WITHOUT resize, so crops keep full resolution);
  2. crop every damage_region box with padding (fraction of box width/height on each side, clipped);
  3. name crops exactly like the labelled set:  <original-photo ID>__image-<image id>__ann-<ann id>.jpg
     so grouping by original photo keeps working;
  4. EXCLUDE every original photo already present in the labelled dataset (labelled_manifest), so no crop of a
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
