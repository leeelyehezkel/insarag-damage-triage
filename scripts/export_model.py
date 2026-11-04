"""Package trained runs as a release folder (for GitHub releases / Hugging Face / Zenodo).

    python scripts/export_model.py --runs <runs/final/v3_s_partial_448> --out release/insarag-triage-dinov3-vits16

Copies every seed's model.pt, writes config.json (read by insarag_triage.inference.load_model), the DINOv3 License
(required with DINOv3-derived weights) and a model card. Validation/test metrics are copied from each seed's result.json.
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", required=True, help="folder with seed*/model.pt and seed*/result.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()
runs, out = Path(a.runs), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
seeds = sorted(p for p in runs.glob("seed*") if (p / "model.pt").exists())
if not seeds:
    raise SystemExit(f"no seed*/model.pt under {runs}")
results = [json.loads((s / "result.json").read_text()) for s in seeds]
c = results[0]["config"]
if c.get("mode") == "frozen":
    raise SystemExit("frozen runs store only the head; export a fine-tuned run (model.pt with the full network)")
weights = []
for s in seeds:
    f = f"model_{s.name}.pt"
    shutil.copy(s / "model.pt", out / f)
    weights.append(f)
cfg = {"name": out.name, "backbone": c["backbone"], "img_size": c["img_size"], "head": c.get("head", "linear"),
       "mode": c["mode"], "grades": ["A", "B", "C", "D", "E", "F"], "weights": weights,
       "loss": "CORN ordinal (rank-consistent)", "preprocessing": "RGB, letterbox to img_size (no stretching), backbone normalisation",
       "per_seed_metrics": [{"seed": r["seed"], "val_overall_accuracy": r["val"]["overall_accuracy_diag_over_6"],
                             "test1_overall_accuracy": r["test"]["overall_accuracy_diag_over_6"],
                             "test1_mae": r["test"]["mae_overall_micro"]} for r in results]}
(out / "config.json").write_text(json.dumps(cfg, indent=2))
repo = Path(__file__).resolve().parents[1]
shutil.copy(repo / "licenses" / "DINOv3-LICENSE.md", out / "LICENSE-DINOv3.md")
card = repo / "docs" / "MODEL_CARD.md"
if card.exists():
    shutil.copy(card, out / "README.md")
print(f"release written to {out}: {len(weights)} seeds · {cfg['backbone']} @ {cfg['img_size']} px")
