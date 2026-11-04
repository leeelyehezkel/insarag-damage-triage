"""Build the public label manifest from the project's working files (run once by the dataset author).

    python scripts/export_manifests.py --work <working directory of the original notebooks> --out data/manifests

Inputs (all inside --work):
    al_labelled_native.csv        every labelled crop: path, label, split (train/val/test), al_round
    runs/final/final_split.csv    crop id → split of the final models (adds the pool validation set)
    leakage.csv                   crops identical to a test crop (their training copies are marked 'excluded')
    blind/blind_labels.csv        blind re-labels of test 1, and test 2
    test3/test3_labels.csv        test 3

Output: labels.csv with one row per crop and NO local paths:
    crop_id, photo, label, split, subset, al_round, source[, label_blind]
"""
import argparse
from pathlib import Path

import pandas as pd

from insarag_triage import core as L

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--work", required=True)
ap.add_argument("--out", default="data/manifests")
a = ap.parse_args()
W, out = Path(a.work), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
photo = lambda p: L.NAME_RE.match(Path(str(p)).name)["group"].upper()

df = pd.read_csv(W / "al_labelled_native.csv", dtype={"group": str})
df["crop_id"] = df["path"].map(L.crop_id)
df["photo"] = df["path"].map(photo)
sp = pd.read_csv(W / "runs" / "final" / "final_split.csv").set_index("cid")["split"]
df["split"] = df["crop_id"].map(sp).fillna(df["split"]).replace({"test": "test1"})
lk = pd.read_csv(W / "leakage.csv")
held = set()
for f in (W / "blind" / "test2_photos.txt", W / "test3" / "test3_photos.txt"):
    if f.exists():
        held |= set(f.read_text().split())
excl = (df.split == "train") & (df.crop_id.isin(set(lk.loc[lk.role == "train", "cid"])) | df.photo.isin(held))
df.loc[excl, "split"] = "excluded"
df["subset"] = df["al_round"].map(lambda r: "original" if r == 0 else "active_learning")
df["source"] = df["al_round"].map(lambda r: "original" if r == 0 else "al_labels")

b = pd.read_csv(W / "blind" / "blind_labels.csv", dtype={"group": str})
t1 = b[b.set == "test1"].set_index("crop_id")["new_label"]
df["label_blind"] = df["crop_id"].map(t1)                     # test-1 crops only
t2 = b[(b.set == "test2") & b.new_label.isin(L.CLASSES)]
t2 = pd.DataFrame({"crop_id": t2.crop_id, "photo": t2.group.str.upper(), "label": t2.new_label, "split": "test2",
                   "subset": "random_blind", "al_round": -1, "source": "test12"})
t3 = pd.read_csv(W / "test3" / "test3_labels.csv", dtype={"group": str})
bad3 = set(lk.loc[(lk.role == "eval") & (lk.set == "test3"), "cid"])
t3 = t3[t3.label.isin(L.CLASSES)]
t3 = pd.DataFrame({"crop_id": t3.crop_id, "photo": t3.group.str.upper(), "label": t3.label,
                   "split": ["excluded" if c in bad3 else "test3" for c in t3.crop_id],
                   "subset": "random_blind", "al_round": -1, "source": "test3"})
cols = ["crop_id", "photo", "label", "split", "subset", "al_round", "source", "label_blind"]
m = pd.concat([df[cols], t2, t3], ignore_index=True)
m.to_csv(out / "labels.csv", index=False)
print(m.pivot_table(index="split", columns="label", values="crop_id", aggfunc="count", fill_value=0).reindex(columns=L.CLASSES))
print(f"wrote {out / 'labels.csv'} ({len(m)} rows, no file paths)")
