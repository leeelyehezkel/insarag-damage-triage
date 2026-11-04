# Damage triage of buildings in post-earthquake photos

Code, data and model for grading the damage of single buildings in street-level photos taken after an earthquake. A photo goes through two steps. SAM 3 finds the buildings from text prompts, and each building is cropped. A DINOv3 ViT-S/16 model then gives each crop a grade from A (intact) to F (debris). The grades are adapted from the INSARAG worksite triage categories and describe only what is visible.

The training data were grown with active learning. A committee of five models chose which crops to label next.

| Grade | Meaning |
|---|---|
| A | intact |
| B | intact, broken windows or minor damage |
| C | standing, damaged (cracks) |
| D | collapsed with voids, structure still recognisable |
| E | pancake collapse, stacked layers |
| F | debris |

See [`docs/labelling_guideline.md`](docs/labelling_guideline.md) for the full definitions and borderline rules, and [`docs/method.md`](docs/method.md) for the method and references.

## Notebooks

| Notebook | What it does |
|---|---|
| [`00_finding_buildings`](notebooks/00_finding_buildings.ipynb) | how the SAM 3 prompts and thresholds were chosen; repeat the comparison, try your own prompts, visual check |
| [`01_photos_to_grades`](notebooks/01_photos_to_grades.ipynb) | photos → buildings → crops → grades, with the released model |
| [`02_data_and_splits`](notebooks/02_data_and_splits.ipynb) | the data, the split by photo, un-stretched crops, the quality filter, the three test sets, the leak check, label consistency |
| [`03_train_and_evaluate`](notebooks/03_train_and_evaluate.ipynb) | the grading model (DINOv3, CORN ordinal head, class weights), training, metrics with intervals, comparing settings |
| [`04_active_learning`](notebooks/04_active_learning.ipynb) | the committee and the batch selection; can replay the study's rounds on the public data, against random selection |
| [`05_final_results`](notebooks/05_final_results.ipynb) | final training of all five models, the model choice, test 3, standing vs collapsed, the learning curve |

All notebooks run on Google Colab with a T4 GPU.

## Install

```bash
pip install "insarag-triage[detect,roboflow] @ git+https://github.com/leeelyehezkel/insarag-damage-triage"
```

Grading a folder of crops from the command line:

```bash
python scripts/predict.py --model leeely/insarag-triage-dinov3-vits16 --images my_crops/ --out grades.csv
```

## Data

The crops and labels are public on Roboflow. [`configs/data.yaml`](configs/data.yaml) lists the projects and versions, and `data/manifests/labels.csv` gives each crop's grade, split and source. Downloading requires a free Roboflow API key. The photos are also on Kaggle ([images](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-images), [masks](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-masks)). All are from the February 2023 earthquakes in Türkiye.

| Split | Crops | Notes |
|---|---|---|
| train | 1,620 | 723 original + active-learning and bulk-labelled crops; copies of test crops removed |
| validation | 384 | 169 original + 215 from the active-learning pool (split by photo) |
| test 1 | 158 | original test split |
| test 2 | 193 | random crops, labelled blind; used during development |
| test 3 | 250 drawn | random crops, labelled blind; final evaluation only (2 duplicate photos removed) |

No photo appears in both training and a test set. Identical pictures stored under different names were found with image fingerprints and removed (see `insarag_triage.leakage`).

## Results

**Finding buildings (SAM 3).** Scores are from 52 test photos with 155 hand-drawn buildings, using one-to-one matching at IoU ≥ 0.5. Precision was 0.82 and recall 0.39. Recall was 0.74 for grade C and 0.41 for grade D. Many strict "misses" are one box drawn around a rubble field that the reference splits into several buildings. Counting a building as found when half of it is covered, validation recall is 0.71 instead of 0.45. See notebook 00.

**Grading (DINOv3 ViT-S/16, fine-tuned, 3 seeds averaged).** The model was chosen on validation data before test 3 was opened. Test 3 has 152 gradeable crops, blind labels, and was used once:

| Measure | Test 3 |
|---|---|
| Balanced accuracy | 0.73 [0.66–0.80] |
| MAE (grades) | 0.29 |
| Within one grade | 97% |
| Accuracy C / D | 0.72 / 0.87 |
| Collapsed (D–F) predicted as standing (A–C) | 0 of 67 |

The weakest grade is B (0.47, only 15 crops). All 11 standing/collapsed errors went the cautious way, mostly C predicted as D. On test 2 (193 random crops) balanced accuracy was 0.69, and about 0.69 on test 1 with its original labels.

Labels are not perfectly repeatable. Re-labelling the test-1 crops blind gave the same grade 64% of the time and a grade within one step 94% of the time (quadratic-weighted κ 0.90).

## Repository layout

```
configs/      data sources, SAM 3 prompts and thresholds
data/         label manifest (images are downloaded, not stored here)
docs/         labelling guideline, method, protocol, model card
licenses/     licence texts of this code and of the models it builds on
notebooks/
scripts/      predict.py, export_model.py, export_manifests.py
src/insarag_triage/
tests/
```

## Licences

- **Code:** [PolyForm Noncommercial 1.0.0](LICENSE). Free for research, teaching and other non-commercial use; commercial use is not permitted.
- **Model weights:** derived from DINOv3, so they are distributed under the [DINOv3 License](licenses/DINOv3-LICENSE.md), which comes with the weights.
- **Dataset (labels and crops):** see [`docs/licences.md`](docs/licences.md).
- **SAM 3** is used under Meta's SAM License; publications using it must acknowledge it.

## Citation

If you use the code, data or model, please cite them as given in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

DINOv3 and SAM 3 by Meta AI; labelling done in Roboflow.
