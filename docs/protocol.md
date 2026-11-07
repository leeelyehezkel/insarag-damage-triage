<!-- Pre-registered analysis protocol with dated addenda and deviations, as kept during the study. -->
# Experiment protocol: supervised A–F model (frozen 2026-10-13, before any results)

**Data.** Roboflow `worksite-triage-post-crop-datase`, version **v1**, one earthquake. One label A–F per crop. Ordinal coding: A=0 … F=5 (A and F farthest apart).

**Split** (fixed once, then never regenerated):
- Pool all crops (Roboflow's own split is ignored) and remove duplicates by (group, image, ann).
- `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`:
  - fold 0 → **test**
  - fold 1 → **val**
  - folds 2–6 → **train** (≈71/14/14)
- Group = original-photo UUID (file-name prefix before `__image-`). No group appears in two splits, so test images are unseen photos.
- Saved as `manifest.csv`. Later sessions reuse it.

**Models** (3 configs × seeds 0, 1, 2):

| Config | Backbone (timm) | Trainable | LR | Epochs / stop |
|---|---|---|---|---|
| `dino_s_frozen` | `vit_small_patch14_dinov2.lvd142m` @224 | head only (features cached once) | head 1e-3 | ≤300 head epochs, patience 40 |
| `dino_s_partial` | same | last 4 ViT blocks + final norm + head | head 1e-3, backbone 2e-5 | ≤40, patience 8 |
| `convnextv2_atto_partial` | `convnextv2_atto.fcmae_ft_in1k` @224 | last 2 stages + head | head 1e-3, backbone 1e-4 | ≤40, patience 8 |

**Common settings:**
- **Features:** ViT uses CLS ⊕ mean patch token; CNN uses global pooling.
- **Head:** dropout 0.1 → linear with 5 outputs (CORN).
- **Optimiser:** AdamW, weight decay 0.05, 2 warm-up epochs, then cosine schedule. Batch size 32. Mixed precision on GPU.
- **Input:** letterbox to square (no distortion), 224 px, ImageNet normalisation.
- **Train augmentation:** RandomResizedCrop (scale 0.75–1), horizontal flip, rotation ±8°, brightness/contrast ±0.15. No vertical flip.

**Loss:** CORN ordinal loss (Shi et al. 2023). Each sample is weighted by w_c = N / (6 · n_c), computed on the training split (inverse class prevalence; mean weight 1). There is no weighted sampler, to avoid correcting for imbalance twice.

**Model selection:**
- **Epoch:** the best validation overall accuracy (diagonal ÷ 6), with macro MAE as tie-break.
- **Config:** the highest mean validation overall accuracy over the 3 seeds. Decided on **validation only**.
- **Test:** evaluated once per run at the selected epoch. Test numbers are never used for any choice.

**Reported metrics** (test, mean ± std over seeds, plus the confusion matrix pooled over seeds):
- 6×6 confusion matrix: counts and row-normalised.
- **Overall accuracy** = Σ diagonal of the row-normalised matrix ÷ 6 (= balanced accuracy).
- Per-class accuracy (diagonal entries).
- **MAE** in grade steps:
  - overall micro (all images);
  - overall macro (mean of per-class MAE);
  - per class.
- Plain accuracy over all images, for reference only.

**Deviations** from this protocol are recorded in this file, with a date and a reason.

---

## Addendum 1: improvement round 1 (2026-10-13, written before running it)

**Motivation:** the error gallery of `dino_s_frozen` showed fine A/B/C details lost at 224 px, and structure lost in close-up D/E/F crops.

**New configs**, all frozen, 3 seeds each, same split, loss, weights, head training and early stopping as above:

| Config | Backbone | Input | Head |
|---|---|---|---|
| `dino_s_frozen_448` | ViT-S/14 | 448 | linear |
| `dino_s_frozen_518` | ViT-S/14 | 518 | linear |
| `dino_b_frozen_224` | ViT-B/14 | 224 | linear |
| `dino_b_frozen_448` | ViT-B/14 | 448 | linear |
| `dino_s_frozen_448_mlp` | ViT-S/14 | 448 | MLP: 768→256 → GELU → dropout 0.2 → 5 |

**Selection:**
- unchanged: the highest mean **validation** overall accuracy over the 3 seeds;
- the candidate pool is the 3 original configs plus these 5;
- test is reported for all, never used to choose.

Cross-validation was considered and not used. A validation difference under ~0.07 is therefore within noise and should be reported as such.

**Diagnostic, no model change:** count the EXIF orientation tags in the data. The loader does not apply them.

**Implementation notes:**
- Frozen features are cached per (backbone, resolution) and reused across seeds and heads. This is identical in result to recomputing them, since there is no augmentation in frozen mode.
- Frozen runs save only `head.pt`.

---

## Addendum 2: learning curve (2026-10-13; diagnostic, no model selection)

**Question:** would more labelled data of the same kind improve the frozen-head model?

**Method:**
- Frozen ViT-S features at two resolutions: DINOv3 ViT-S/16 at 224 and 512 px (changed from DINOv2 224/518 by Addendum 3 before the curve was run).
- Draw a random subset of 25 / 50 / 75 / 100 % of the training **photos** (grouped; redrawn until all 6 classes are present). 5 repeats per fraction.
- Train the linear CORN head with the same procedure as the main runs: class weights from the subset, early stopping on validation.

**Reported:** validation overall accuracy (diag/6), macro MAE and per-class accuracy against the number of training crops. **The test set is not used.**

**Reading:**
- Still rising at 100 %: more data is expected to help, so grow the dataset (active learning, targeting the classes still rising).
- Flat from 75 % on: prioritise model changes.

---

## Addendum 3: switch to DINOv3 (2026-10-13, before running)

**Reason:** DINOv3 (Siméoni et al. 2025) was chosen as the backbone family. All DINOv2 results are kept and reported for comparison.

**Weights** (timm, `lvd1689m`, DINOv3 licence): ViT-S/16 (21.6M), ViT-S+/16 (28.7M), ConvNeXt-Tiny distilled (27.8M). DINOv3 uses 16-px patches, so sizes are 224 / 448 / 512.

**Configs** (3 seeds each; same split, loss, weights and training as the main protocol):

| Config | Backbone | Input | Trainable | LR (backbone) |
|---|---|---|---|---|
| `v3_s_frozen_224` | ViT-S/16 | 224 | head | – |
| `v3_s_frozen_448` | ViT-S/16 | 448 | head | – |
| `v3_s_frozen_512` | ViT-S/16 | 512 | head | – |
| `v3_splus_frozen_448` | ViT-S+/16 | 448 | head | – |
| `v3_cnxt_frozen_448` | ConvNeXt-T (DINOv3) | 448 | head | – |
| `v3_s_partial_448` | ViT-S/16 | 448 | last 4 blocks + norm + head | 2e-5 |
| `v3_cnxt_partial_448` | ConvNeXt-T (DINOv3) | 448 | last 2 stages + head | 5e-5 |

**Selection:** the highest mean **validation** accuracy (diag/6) over 3 seeds **among the DINOv3 configs**. Test is reported for all and never used to choose. Validation differences under ~0.07 are within noise.

**Learning curve (Addendum 2):** now run on DINOv3 ViT-S/16 at 224 and 512 px.

---

## Addendum 4: active learning with a query-by-committee (2026-10-13, before round 1)

**Pool:** crops of `damage_region` boxes from Roboflow `damage-region-dataset`, with the padding measured on the labelled crops. Photos already in the labelled set are excluded.

**Committee:** 5 members × 3 seeds. Members chosen for training-paradigm diversity (Gontijo-Lopes 2022; Melville 2004); 5 members per Lakshminarayanan 2017 and Beluch 2018.
- DINOv3 ViT-S partial @448
- DINOv3 ViT-S frozen @224
- DINOv2 ViT-S frozen @224
- DINOv3 ConvNeXt-T partial @448
- SigLIP 2 B/16 frozen @224 (language-supervised)

**Score:**
- For each CORN threshold t, binary BALD: H(mean p_t) − mean H(p_t) over members (each member averaged over its seeds). Ordinal → binary reduction per Haas 2025b; BALD per Houlsby 2011.
- Weights 1 / 2 / 2 / 2 / 1 for A|B, B|C, C|D, D|E, E|F (C/D priority).

**Selection** (100 per round):
- shorter side ≥ 64 px;
- the top 500 by score;
- 70 % to the focus group (committee predicts C or D, or the most-disputed threshold is B|C, C|D or D|E), 30 % to the rest;
- within each group, k-means on DINOv3 ViT-S @224 embeddings, taking the crop nearest each centre;
- ≤ 2 crops per original photo.

**Labelling:** only the selected crops are uploaded to Roboflow `damage-level-classification`, batch "AL round k", tagged with `al_round_k` and the committee prediction. Labelled crops go to **train only**. Val/test are frozen; crops from val/test photos are refused.

**Evaluation per round:** every member's val and test overall accuracy, MAE, and C and D accuracy (`round_table.csv`). Round 1 (original data) is the baseline.

**Stopping:** decided by the author. The rule used is to stop when C/D validation accuracy stops improving for 2 rounds.

**Limitations:**
- No random-sampling control arm within the real labelling budget. AL-vs-random evidence comes from a simulation on the existing labels (notebook 04, `ARM = 'random'`).
- The selected crops are by design not representative, so per-round gains are measured only on the frozen val/test.

**Amendment to Addendum 4** (2026-10-13, after viewing the round-1 selection and **before any upload or labelling**). The preview showed non-building crops (street panoramas, people, interiors, tarps), slivers, black or white crops, and the same building twice from overlapping boxes. Changes:
- **Pool clean-up:** within a source image, overlapping boxes (IoU ≥ 0.5, or ≥ 80 % of the smaller box inside the larger) → keep the larger one.
- **Quality filter:** aspect ≤ 3:1; shorter side ≥ 96 px; mean luminance 0.12–0.90; luminance std ≥ 0.05.
- **Ranking:** committee score × normalised density, where density is the mean cosine similarity to the 20 nearest pool neighbours (Kee 2018). This prevents outliers from dominating.
- **Photo cap:** ≤ 1 crop per original photo per round.
- **Roboflow class `exclude`:** for crops that are not a gradable building. Ignored at ingest, never used for training.

---

## Addendum 5: stand-alone active-learning notebook and warm start (2026-10-14, from round 3)

**Goal:** the best model as quickly as possible, rather than a controlled study of data effects.

**From round 3:**
- **Fine-tuned members:** continue from the previous round's weights (same seed), at most 10 epochs, 1 seed (0), early stopping on validation unchanged.
- **Frozen members:** retrain from scratch with 3 seeds (seconds).
- **Rationale:** speed. Warm-starting can generalise slightly worse than fresh training (Ash & Adams 2020), so the reported final model is retrained **from scratch** on all labels with 3 seeds (notebook section 7).

**Uploads:** pre-labelled with the committee prediction as the Roboflow class (classification folder import); fallback is an unlabelled upload with `pred_X` tags.

**Tracking:** per member and round, on frozen val/test: overall accuracy, MAE (micro/macro), per-class accuracy and MAE. Plotted against the number of training crops. Batch reports compare each member's pre-labelling guess with the author's label.

**Amendment to Addendum 5** (2026-10-14, before round 3):
- **Warm start:** keeps the backbone and re-initialises the head (Nikishin 2022; Kumar 2022).
- **Cold rounds:** every 3rd round trains from scratch (Dohare 2024; Omar 2026).
- **Diversity:** pairwise disagreement and error consistency (Cohen's kappa of right/wrong; Geirhos 2021; Kuncheva 2003) are reported per round. Any change of committee member is decided on these measurements and logged before the round that uses it.

**Committee change** (2026-10-14, **before round 3**): `dino_s_frozen` (DINOv2 ViT-S, frozen) is replaced by `cnx_in22k_frozen_224`: ConvNeXt-Tiny supervised on ImageNet-22k (`convnext_tiny.fb_in22k`), frozen, 224 px, 3 seeds.
- **Rationale:** DINOv2 shares its pretraining objective and family with the DINOv3 members and is likely redundant. The replacement adds a third pretraining objective (supervised labels) and a CNN not distilled from DINO (Gontijo-Lopes 2022; Mustafa 2020; Zaidi 2021).
- **Backfill:** the new member is trained on round 1's and round 2's exact training sets, so its curve starts at round 1. DINOv2's rounds 1–2 stay in the tracking plots as a retired member.
- **Check:** the round-3 diversity table (disagreement, error consistency) verifies the change.

## Addendum 6: end of AL rounds, blind re-labelling and test set 2 (2026-10-18, before labelling)

**AL stopped after round 5** (rounds 1–5; ~1,068 training crops, +345 over the baseline). Stop rule (C/D flat for 2 rounds) met.
- Paired cluster bootstrap, round 1 → 5, same test crops: no member changed significantly. Best: `v3_s_partial_448` +0.019 [−0.049, +0.076].
- Committee median vote 0.624 vs best member 0.640, not significant. The committee is kept for selection only, not for prediction.
- 17 of 158 test crops are wrong for all 5 members.
- Test support is A 30, B 20, C 12, D 30, E 21, F 45. C accuracy moves in steps of 0.083, so the test set cannot resolve C.

**Blind re-labelling (label consistency):** the author re-labels all 158 test crops in a separate Roboflow project, with no labels, predictions or tags visible.
- Mixed in random order with test 2.
- Reported: exact agreement, within-1, mean absolute difference, Cohen's kappa and quadratic-weighted kappa, per-grade self-agreement and the confusion matrix.
- The **official test-1 labels stay the original ones**. Blind labels are reported alongside, never substituted selectively.

**Test set 2:** 200 crops drawn at random (seed 2026) from the quality-filtered pool.
- One crop per photo. Only photos never labelled or selected by AL in any round.
- Labelled blind; these photos are removed from the AL pool permanently.
- Reported for every member, latest round, with 95% cluster-bootstrap CIs. Also reported pooled with test 1 (blind labels).

**Final model:** `v3_s_partial_448`, trained from scratch on all labels. Reported on test 1 (original labels), test 1 (blind labels) and test 2.

**Quality rule for labelled data:** to be decided after the blind results. The rule is the pool's thresholds (aspect ≤ 3, short side ≥ 96 px, luminance 0.12–0.90, std ≥ 0.05). If adopted, it is applied equally to train, val and test, and the change is logged here before any rerun.

**Amendment to Addendum 6** (2026-10-18, before any blind labels were used): the 1,050 original crops, from Roboflow v1 of the main project, are all 224×224, stretched to a square. All AL and pool crops are full resolution at their natural aspect.
- Every result so far mixes stretched originals with un-stretched AL crops, and the test set itself is stretched.
- **Fix:** a new Roboflow version with Resize off replaces the originals; names, labels and the frozen split are unchanged.
- **Check:** the best model (`v3_s_partial_448`) is retrained from scratch, 3 seeds, on (a) the original 723 training crops and (b) all 1,068, both un-stretched. It is compared with the stretched results (0.620 and 0.640).
- **Blind batch:** redone in a new Roboflow project with the un-stretched test-1 crops. Labels from the first blind project are discarded.

## Addendum 7: results of the blind re-labelling and the un-stretched retraining (2026-10-20, recorded after the results)

**Label consistency** (test 1: 144 graded crops; 14 more marked exclude).
- Exact agreement 0.639; within one grade 0.938.
- Cohen's κ 0.562; quadratic-weighted κ 0.900.
- Per grade: A 0.42, B 0.41, C 0.91, D 0.78, E 0.67, F 0.68.
- The changes are directional: standing crops moved up (A→B 9, A→C 5, B→C 9) and collapsed crops moved down (F→E 10, E→D 5, F→D 3).
- No image property explained the changes (all p > 0.1). Exclusion was explained by size: crops under 96 px were excluded 43% of the time vs 5% (p = 0.005).

**Un-stretched model** (`v3_s_partial_448`, from scratch, 3 seeds, all 1,068 training crops).
- Test 1, original labels: 0.672, MAE 0.333 (stretched, round 5: 0.640, MAE 0.354).
- Test 2 (193 random graded crops): 0.652 [0.580, 0.720], MAE 0.440.
- The same model trained without the AL crops reaches 0.573 [0.505, 0.638], MAE 0.751 on test 2.
- Test 1 + test 2, blind labels (337 crops, 82 C): 0.659 [0.606, 0.709], MAE 0.412, C 0.549 [0.438, 0.655].

**Decisions (confirmed before the final training):**
1. All further work uses un-stretched crops.
2. Quality rule: short side ≥ 96 px, applied equally to train, val and test. Evidence: the annotator excluded such crops 9× more often.
3. Test 1 keeps its original labels as the official reference. Test 2's blind labels are its only labels.
4. The written decision rules (`docs/labelling_guideline.md`) become the labelling guideline. Disagreements in the training data at the A/B/C and D/E/F boundaries are reviewed before the final model.

## Addendum 8: active learning resumed on un-stretched crops (2026-10-28)

The stop after round 5 (Addendum 6) was made on the stretched data and on test 1, which has only 12 C crops. With un-stretched crops and test 2 in place, the AL crops measurably helped (Addendum 7: 0.573 → 0.652 on test 2), so active learning was resumed.
- Rounds 6–9: the same committee and selection, on un-stretched crops; round 6 trained from scratch.
- Progress is tracked on test 2 from round 6 on.
- Round 10 (2026-11-03): after test 3 was drawn, the rest of the usable pool was added to round 10's batch and labelled in one pass (bulk labelling), pre-filled with the committee's grade.

## Deviation found 2026-11-03: test/validation crops with identical copies in training

A fingerprint check (16×16 average hash) across all crops found **32 test crops (30 in test 2, 2 in test 3) with an identical picture in training or in the round-10 batch.** Validation and test 1 had none. There are two causes:
1. **Test-2 crops picked into training by later AL rounds.** Same crop name. In one working session the test-2 photos were not excluded from the pool, so active learning picked some of them.
2. **The same photo stored under two names** in the source dataset (e.g. `X` and `X__1_`). The photo-wise split cannot catch this.

**Consequence:** test-2 results for rounds 6–10 are optimistic.

**Handling (decided before the final training):**
- The leaked crops are listed in `leakage.csv`.
- The round history and the paired round 6 → 10 test are recomputed on **test 2 minus the leaked crops**.
- **Every training crop identical to any val/test crop is left out of the final training.**
- Test 3 is evaluated without its leaked crop(s).
- From then on, held-out photos are excluded from the pool by name and by fingerprint.

**Leakage results (2026-11-03):**
- 30 test-2 crops had copies in training, added by the AL rounds 6–8 (one further copy, from round 2, is the same photo under another name). By crop id, 34 test-2 crops had been picked into training. The 2 test-3 crops are duplicate photos.
- **Clean test 2 (165 crops), paired round 6 → 10:**
  - Only `v3_cnxt_partial_448` improved significantly overall: +0.119 [+0.034, +0.206], C +0.245, MAE −0.158.
  - `v3_s_partial_448` improved on C (+0.102) and MAE (−0.067), but not significantly overall (+0.046 [−0.010, +0.104]).
  - The frozen models' +0.03 to +0.05 is not significant.
  - D is unchanged or lower for all models.
- This replaces the earlier "all five significant" result.

**Final evaluation plan (fixed before training):**
- All five models are trained cold, 3 seeds, without the leaked training copies and without any crop from a test-2/test-3 photo (73 training crops left out in total).
- Validation = original val + 20% of the AL crops, split by photo (StratifiedGroupKFold, seed 42). The model is chosen on validation; test 2 (all 193, clean for these models) confirms.
- Test 3 (blind, minus 2 duplicate-photo crops) is opened once for the pre-declared choice.

## Deviation 2026-11-04: final choice by C/D, not by overall validation accuracy (before test 3 was opened)

**Final comparison (validation = 20% AL crops + original val; test 2 = 193 blind crops):**

| Model | val acc [95% CI] | val MAE | test-2 acc [95% CI] | test-2 MAE | test-2 C | test-2 D |
|---|---|---|---|---|---|---|
| ENSEMBLE (2 fine-tuned) | 0.767 [0.721, 0.810] | 0.305 | 0.702 [0.632, 0.772] | 0.394 | 0.704 | 0.595 |
| v3_cnxt_partial_448 | 0.737 [0.689, 0.782] | 0.346 | 0.702 [0.635, 0.772] | 0.430 | 0.704 | 0.595 |
| v3_s_partial_448 | 0.718 [0.669, 0.764] | 0.354 | 0.691 [0.617, 0.762] | 0.383 | 0.685 | 0.714 |
| v3_s_frozen_224 | 0.691 [0.641, 0.739] | 0.411 | 0.645 [0.574, 0.715] | 0.482 | 0.630 | 0.548 |
| siglip2_frozen_224 | 0.664 [0.619, 0.710] | 0.466 | 0.655 [0.583, 0.728] | 0.477 | 0.685 | 0.524 |
| cnx_in22k_frozen_224 | 0.592 [0.541, 0.641] | 0.560 | 0.615 [0.544, 0.687] | 0.487 | 0.667 | 0.571 |

**Pre-declared rule** (highest validation accuracy) would select `ENSEMBLE (2 fine-tuned)`.

**Decision:** the author chooses **`v3_s_partial_448`** (DINOv3 ViT-S/16, last 4 blocks fine-tuned, 448 px), to optimise for the priority grades C and D.
- The top three validation CIs overlap.
- On test 2, `v3_s_partial_448` has the best C/D mean: (0.685 + 0.714) / 2 = 0.70, vs 0.65 for the ensemble and the ConvNeXt. It also has the lowest MAE (0.383).
- It is a single model: half the inference cost of the ensemble, which matters for the phone target.

**Caveat:** this choice used test-2 per-grade results, so test 2 is no longer a pure confirmation set for the chosen model. Test 3 was not opened before the choice and remains the unbiased final estimate. Test-3 results of the other models are not used to revisit the choice.

## Final result: test 3, opened once (2026-11-04)

**Model:** `v3_s_partial_448`, the pre-declared choice (see the deviation above), 3 seeds averaged.
**Data:** test 3, blind labels, duplicate-photo crops removed. 152 graded crops: A 17, B 15, C 53, D 30, E 18, F 19 (counts from the confusion matrix).

| Metric | Estimate | 95% CI (cluster bootstrap, 2,000) |
|---|---|---|
| Balanced accuracy (diag/6) | **0.733** | [0.660, 0.803] |
| MAE micro | 0.289 | [0.211, 0.375] |
| MAE macro | 0.298 | [0.212, 0.388] |
| Within one grade (balanced) | 0.973 | [0.941, 0.997] |

| Grade | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| Accuracy | 0.941 | 0.467 | 0.717 | 0.867 | 0.667 | 0.737 |
| 95% CI | [0.800, 1.000] | [0.200, 0.727] | [0.589, 0.833] | [0.731, 0.971] | [0.435, 0.889] | [0.526, 0.933] |
| MAE | 0.118 | 0.533 | 0.321 | 0.167 | 0.333 | 0.316 |

- Most frequent confusions: C→D 10, E→D 5, then B→A, B→C, C→B and F→E with 4 each. All but 4 errors are one grade off (A→C 1, C→F 1, D→F 1, F→D 1).
- B remains the weakest grade (15 crops, wide CI).
- The priority grades are C 0.72 and D 0.87.
- The test-3 scores of the other models were shown for reference after the choice. They are not used to revisit it.
- Test 3 is now spent. Any later change to the model needs a new held-out set, for example photos from a different earthquake.

## References cited in this protocol

Melville, P., & Mooney, R. J. (2004). Diverse ensembles for active learning. ICML · doi:10.1145/1015330.1015385 ·
Haas, S., & Hüllermeier, E. (2025b). Aleatoric and epistemic uncertainty measures for ordinal classification through binary reduction. arXiv:2507.00733 ·
Omar, N., Rozenshtein, M., Mishlyakov, E., & Gal, A. (2026). Train smarter, not harder: switching signal-guided training in active learning. arXiv:2609.06806 ·
All other references are listed in `docs/method.md`.

