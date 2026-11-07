---
license: other
license_name: dinov3-license
license_link: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
library_name: pytorch
pipeline_tag: image-classification
tags:
  - earthquake
  - building-damage
  - dinov3
  - ordinal-classification
---

# insarag-triage DINOv3 ViT-S/16

Grades a crop of a single building in a post-earthquake street-level photo from A (intact) to F (debris). Code and documentation: https://github.com/leeelyehezkel/insarag-damage-triage

- **Architecture:** DINOv3 ViT-S/16 (21.6M parameters). The last 4 blocks are fine-tuned, with a CORN ordinal head and 448 px letterboxed input.
- **Weights:** three training seeds (`model_seed*.pt`); predictions are the average over seeds.
- **Training data:** 1,620 labelled crops from the February 2023 earthquakes in Türkiye.
- **Intended use:** research on post-earthquake damage assessment and rescue prioritisation. The model hasn't been tested on other events, building types or aerial images. It must not replace an on-site assessment.
- **Known limits:** most errors are between neighbouring grades. Grades A and B are also the least consistent in human labelling (a blind re-labelling repeated 42% and 41% of them).
- **License:** DINOv3 License (`LICENSE-DINOv3.md`, included with the weights). Please cite as given in the repository.

```python
from insarag_triage.inference import load_model, predict
model = load_model("leeely/insarag-triage-dinov3-vits16")
grades = predict(model, ["crop1.jpg", "crop2.jpg"])
```
