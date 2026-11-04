import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from insarag_triage import core as L
from insarag_triage import detect as D
from insarag_triage.inference import load_model, predict
from insarag_triage.leakage import find_leaks


def _img(path, size=(160, 120), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8")).save(path)
    return path


def test_metrics_balanced_accuracy():
    y = np.array([0, 0, 1, 2, 3, 4, 5, 5])
    m = L.evaluate(y, y)
    assert m["overall_accuracy_diag_over_6"] == 1.0 and m["mae_overall_micro"] == 0.0


def test_corn_predict_is_rank_consistent():
    logits = torch.tensor([[5.0, 5.0, -5.0, -5.0, -5.0]])
    assert L.corn_predict(logits).tolist() == [2]


def test_release_roundtrip(tmp_path):
    model = L.TriageNet("dinov3_vits16", 224, pretrained=False)
    torch.save(model.state_dict(), tmp_path / "model_seed0.pt")
    (tmp_path / "config.json").write_text(json.dumps(
        {"backbone": "dinov3_vits16", "img_size": 224, "head": "linear", "weights": ["model_seed0.pt"]}))
    rel = load_model(tmp_path, device="cpu")
    paths = [_img(tmp_path / f"c{i}__image-0__ann-{i}.jpg", seed=i) for i in range(3)]
    out = predict(rel, paths, num_workers=0)
    assert len(out) == 3 and set(out.grade) <= set(L.CLASSES)
    probs = out[[f"P({c})" for c in L.CLASSES]].to_numpy()
    assert np.allclose(probs.sum(1), 1.0) and (probs >= -1e-6).all()


def _dets():
    return pd.DataFrame([
        dict(photo="p.jpg", prompt="building", score=0.9, x=0, y=0, w=400, h=300, img_w=1000, img_h=800),
        dict(photo="p.jpg", prompt="damaged standing building", score=0.6, x=5, y=5, w=390, h=290, img_w=1000, img_h=800),
        dict(photo="p.jpg", prompt="building", score=0.45, x=600, y=100, w=200, h=200, img_w=1000, img_h=800),
        dict(photo="p.jpg", prompt="building", score=0.8, x=900, y=700, w=20, h=20, img_w=1000, img_h=800),
    ])


def test_select_thresholds_nms_and_size():
    thr = {"building": 0.5, "damaged standing building": 0.5}
    size = dict(min_width=0.04, min_height=0.05, min_area=0.004)
    out = D.select(_dets(), thr, nms_iou=0.4, size=size)
    # duplicate removed by NMS, low-score box removed by threshold, tiny box removed by size
    assert len(out) == 1 and out.iloc[0].score == 0.9


def test_score_hungarian_and_f2():
    gt = pd.DataFrame([dict(photo="p.jpg", x=0, y=0, w=400, h=300, grade="C", img_w=1000, img_h=800),
                       dict(photo="p.jpg", x=600, y=100, w=200, h=200, grade="D", img_w=1000, img_h=800)])
    s = D.score(D.select(_dets(), {"building": 0.4, "damaged standing building": 0.4}, nms_iou=0.4), gt)
    assert s["tp"] == 2 and s["fn"] == 0 and s["fp"] == 1 and s["recall_C"] == 1.0
    r = D.region_scores(gt.iloc[:1], gt)
    assert r["found"] == 0.5 and r["area_precision"] == 1.0


def test_find_leaks_across_names(tmp_path):
    a = _img(tmp_path / "X__image-1__ann-1.jpg", seed=1)
    b = tmp_path / "X__1___image-1__ann-1.jpg"
    Image.open(a).save(b)
    c = _img(tmp_path / "Y__image-2__ann-1.jpg", seed=2)
    leaks = find_leaks({"train": [a], "test": [b, c]}, eval_sets=["test"], train_sets=["train"])
    assert set(leaks.file) == {a.name, b.name}
