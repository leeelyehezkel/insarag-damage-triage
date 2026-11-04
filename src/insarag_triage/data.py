"""Get the public dataset: download the Roboflow versions listed in configs/data.yaml and assemble them with the
label manifest (data/manifests/labels.csv), which says for every crop its grade, split and source."""
from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

import pandas as pd

from . import core as L


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def api_key() -> str:
    """Roboflow API key from ROBOFLOW_API_KEY, or asked for once (never written to disk)."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        key = getpass("Roboflow API key (free account; Settings → API Keys): ")
        os.environ["ROBOFLOW_API_KEY"] = key
    return key


def download(cfg: dict, source: str, root: str | Path = "data/roboflow", fmt: str = "folder") -> Path:
    """Download one source into root/<source>-v<version>; skipped if present. fmt = 'folder' (classification:
    class sub-folders) or 'coco' (detection boxes)."""
    s = cfg["sources"][source]
    if s.get("version") is None:
        raise ValueError(f"configs/data.yaml: no version set for source {source!r}")
    out = Path(root) / f"{source}-v{s['version']}"
    if not out.exists():
        from roboflow import Roboflow
        Roboflow(api_key=api_key()).workspace(cfg["workspace"]).project(s["project"]).version(s["version"]).download(
            fmt, location=str(out))
    return out


def index_images(folder: str | Path) -> dict[str, str]:
    """crop id → image path, for every image under folder (Roboflow renames files; the crop id survives)."""
    return {L.crop_id(p.name): str(p) for p in Path(folder).rglob("*")
            if p.suffix.lower() in L.IMG_EXT and L.NAME_RE.match(p.name)}


def assemble(manifest: str | Path | pd.DataFrame, folders: dict[str, str | Path]) -> pd.DataFrame:
    """Join the label manifest with downloaded images. folders: source name → downloaded folder.
    Returns the manifest with a `path` column and `y` (0–5); rows whose image was not found are reported and dropped."""
    m = pd.read_csv(manifest, dtype={"photo": str}) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    idx = {s: index_images(f) for s, f in folders.items()}
    m["path"] = [idx.get(s, {}).get(c) for s, c in zip(m["source"], m["crop_id"])]
    missing = m["path"].isna() & m["source"].isin(list(folders))
    if missing.any():
        print(f"{int(missing.sum())} crops not found in the downloaded folders, e.g. {m.loc[missing, 'crop_id'].head(3).tolist()}")
    m = m[m["path"].notna()].copy()
    m["y"] = m["label"].map({c: i for i, c in enumerate(L.CLASSES)})
    m["group"] = m["photo"]
    return m.reset_index(drop=True)


GT_GRADES = {f"Damage_Level_{g}": g for g in L.CLASSES}


def load_coco_boxes(split_dir: str | Path) -> pd.DataFrame:
    """Hand-drawn building boxes from a Roboflow COCO export split: photo, path, x, y, w, h, grade, img_w, img_h.
    Only the six grade classes are kept (machinery, people and other scene objects are left out)."""
    import json
    split_dir = Path(split_dir)
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    rows = []
    for a in coco["annotations"]:
        grade = GT_GRADES.get(cats[a["category_id"]])
        if grade is None:
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        rows.append(dict(photo=Path(im["file_name"]).name, path=str(split_dir / im["file_name"]), x=x, y=y, w=w, h=h,
                         grade=grade, img_w=im["width"], img_h=im["height"]))
    return pd.DataFrame(rows)
