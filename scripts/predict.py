"""Grade every image in a folder with the released model.

    python scripts/predict.py --model <release folder or Hugging Face repo id> --images <folder> --out predictions.csv
"""
import argparse
from pathlib import Path

from insarag_triage.core import IMG_EXT
from insarag_triage.inference import load_model, predict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", default="predictions.csv")
ap.add_argument("--batch-size", type=int, default=16)
a = ap.parse_args()
paths = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXT)
if not paths:
    raise SystemExit(f"no images found under {a.images}")
res = predict(load_model(a.model), paths, batch_size=a.batch_size)
res.to_csv(a.out, index=False)
print(f"graded {len(res)} crops → {a.out}")
print(res["grade"].value_counts().sort_index().to_string())
