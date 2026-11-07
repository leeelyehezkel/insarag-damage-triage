# Method

This page describes the pipeline from a street-level photo to an A–F grade for each building in it. It covers how the models were trained, and how the active-learning batches were chosen. References are listed at the end.

## 1. Finding and cropping buildings

A single detector trained to find buildings and classify them in one step (RF-DETR) found only 24% of the buildings, so finding and grading were separated. Buildings are found with SAM 3 [Carion et al. 2025] using four text prompts, each with its own score threshold:

| Prompt | Threshold |
|---|---|
| building reduced to rubble | 0.60 |
| ruins of a collapsed building | 0.55 |
| damaged standing building | 0.50 |
| building | 0.50 |

- **Choosing the prompts.** They were chosen on 54 validation photos with hand-corrected boxes. Twenty single phrases were scored, then 1,350 combinations of up to three phrases. About 10,000 settings of per-phrase thresholds and duplicate removal followed, and four phrases for standing buildings were added along the way. Scoring used one-to-one Hungarian matching at IoU ≥ 0.5, with F2 as the main score. Collapse phrases use stricter thresholds because they also match amorphous debris. `building` was added for labelling, where a missed building costs more than an extra box. The full account is in `notebooks/00_finding_buildings.ipynb`.
- **Selecting and cropping boxes.** Boxes found by more than one prompt are merged by NMS at IoU 0.40. Boxes smaller than 4% of the photo width, 5% of its height or 0.4% of its area are dropped. In the study, aerial photos were removed by a small vision-language model (Qwen 3.5 VL 0.8B). Each remaining box is cropped with 10% padding. Crops with a short side under 96 px, or that are very dark, very bright or nearly blank, are not graded.

## 2. The grading model

- **Backbone:** DINOv3 ViT-S/16 [Siméoni et al. 2025], pretrained, input 448 px. The image is padded to a square ("letterbox") and never stretched.
- **Fine-tuning:** the last 4 of the 12 transformer blocks are trained; the rest stay frozen.
- **Feature:** the CLS token concatenated with the mean of the patch tokens.
- **Head and loss:** CORN, an ordinal head with five "grade > t" tasks [Shi et al. 2023]. Its predictions are rank-consistent (if P(y > C) is high, P(y > B) is at least as high). Each training crop's loss is weighted by N / (6 · n_class), so rare grades count as much as common ones.
- **Training:** AdamW with warm-up and a cosine schedule. The epoch with the best validation balanced accuracy is kept, with early stopping after 8 epochs without improvement. Three seeds are trained and their predictions averaged.
- **Metrics:** confusion matrix; balanced accuracy = mean of the per-grade accuracies (sum of the row-normalised diagonal / 6); per-grade accuracy; MAE in grade steps (overall and per grade); within-one-grade accuracy. The 95% intervals come from a bootstrap over photos.

## 3. Choosing the next crops to label (query-by-committee)

Every round, a committee of five models graded every unlabelled crop. The 100 crops the committee disagreed on most informatively were sent for labelling.

**Why a committee.** Query-by-committee picks examples on which several models trained on the same labels disagree [Seung et al. 1992; Freund et al. 1997]. Ensembles of deep networks give better uncertainty estimates than a single network [Lakshminarayanan et al. 2017], and work well for active learning on images [Beluch et al. 2018]. The members were chosen to differ in what they learned before seeing our data:

| Member | Pretraining | Training here |
|---|---|---|
| DINOv3 ViT-S/16, 448 px | self-supervised [Siméoni et al. 2025] | last 4 blocks fine-tuned, 1 seed per round |
| DINOv3 ViT-S/16, 224 px | self-supervised | frozen backbone, head only, 3 seeds |
| DINOv3 ConvNeXt-Tiny, 448 px | distilled from the DINOv3 teacher | last 2 stages fine-tuned, 1 seed per round |
| ConvNeXt-Tiny, ImageNet-22k, 224 px | supervised class labels | frozen backbone, head only, 3 seeds |
| SigLIP 2 B/16, 224 px | image–text pairs | frozen backbone, head only, 3 seeds |

Models that differ in pretraining data and objective make more different errors than models that differ only in architecture or settings [Gontijo-Lopes et al. 2022; Mustafa et al. 2020; Zaidi et al. 2021]. In rounds 1–2 a frozen DINOv2 ViT-S [Oquab et al. 2023] was the fourth member. It was replaced by the supervised ConvNeXt from round 3, because it was too close to the DINOv3 members. Diversity was checked every round as the pairwise disagreement and the error consistency between members (Cohen's kappa on right/wrong) [Geirhos et al. 2021; Kuncheva & Whitaker 2003].

**Step 1: disagreement at each grade boundary.** Each member gives P(y > t) for the five boundaries A|B … E|F, averaged over its seeds. For each boundary, the score is the BALD quantity [Houlsby et al. 2011]: the entropy of the committee's mean probability minus the mean entropy of the members. It is high only when the members are each fairly confident but disagree with one another. A crop that confuses every member equally (a blurry crop, for instance) scores low. The five boundary scores are added with weights 1, 2, 2, 2, 1, so disagreement at B|C, C|D and D|E (the boundaries that decide C and D) counts double.

**Step 2: typicality.** The score is multiplied by the crop's density: its mean cosine similarity to its 20 nearest neighbours in the pool, using frozen DINOv3 features. This keeps rare outliers from filling the batch [Settles 2012; Kee et al. 2018].

**Step 3: uncertain first, then diverse.** The 500 highest-scoring crops that pass the quality filter are the candidates. They are split into a C/D group (the committee predicts C or D, or the strongest disagreement is at B|C, C|D or D|E) and the rest. The C/D group gets 70 of the 100 places and the rest 30. Within each group the candidates are clustered with k-means into as many clusters as places, and the crop closest to each cluster centre is taken. At most one crop per photo is allowed in a batch. Taking the most uncertain crops first and then spreading the picks over the feature space follows the structure of DropQuery [Gupte et al. 2024] and the typicality idea of TypiClust [Hacohen et al. 2022]. It also avoids the redundant batches that pure top-k uncertainty produces [Kirsch et al. 2019; Ash et al. 2020; Sener & Savarese 2018].

**Step 4: labelling.** The 100 crops were uploaded to Roboflow with the committee's grade filled in, and the grades were then corrected by hand.

**Retraining between rounds.** The frozen members are retrained from scratch every round, which takes seconds. To save time, the fine-tuned members continue from the previous round's weights. The head is re-initialised [Nikishin et al. 2022; Kumar et al. 2022], and they train for at most 10 epochs. Because warm-started networks can end up slightly worse than freshly trained ones [Ash & Adams 2020; Dohare et al. 2024], every third round is trained from scratch. The reported final models are always trained from scratch.

## 4. Test sets

| Set | Crops | Labels | Role |
|---|---|---|---|
| Test 1 | 158, from the original dataset (photo-wise split) | original labels; also re-labelled blind | comparison with the original dataset |
| Test 2 | 193, random pool crops, one per photo | blind | tracking the rounds, choosing the model |
| Test 3 | 250, random pool crops, one per photo, drawn before the final labelling | blind | final result, opened once |

No photo of a test set appears in training. Images are also compared by fingerprint, because the same photo can exist under two names (see `insarag_triage.leakage`).

## References

- Ash, J. T., & Adams, R. P. (2020). On warm-starting neural network training. arXiv:1910.08475.
- Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., & Agarwal, A. (2020). Deep batch active learning by diverse, uncertain gradient lower bounds. arXiv:1906.03671.
- Beluch, W. H., Genewein, T., Nürnberger, A., & Köhler, J. M. (2018). The power of ensembles for active learning in image classification. CVPR, 9368–9377. doi:10.1109/CVPR.2018.00976
- Carion, N., et al. (2025). SAM 3: Segment Anything with Concepts. arXiv:2511.16719 (ICLR 2026).
- Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual learning. Nature, 632, 768–774. doi:10.1038/s41586-024-07711-7
- Freund, Y., Seung, H. S., Shamir, E., & Tishby, N. (1997). Selective sampling using the query by committee algorithm. Machine Learning, 28, 133–168. doi:10.1023/A:1007330508534
- Geirhos, R., et al. (2021). Partial success in closing the gap between human and machine vision. arXiv:2106.07411.
- Gontijo-Lopes, R., Dauphin, Y., & Cubuk, E. D. (2022). No one representation to rule them all: Overlapping features of training methods. arXiv:2110.12899.
- Gupte, S. R., Aklilu, J., Nirschl, J. J., & Yeung-Levy, S. (2024). Revisiting active learning in the era of vision foundation models. arXiv:2401.14555.
- Hacohen, G., Dekel, A., & Weinshall, D. (2022). Active learning on a budget: Opposite strategies suit high and low budgets. arXiv:2202.02794.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745.
- Kee, S., del Castillo, E., & Runger, G. (2018). Query-by-committee improvement with diversity and density in batch active learning. Information Sciences, 454–455, 401–418. doi:10.1016/j.ins.2018.05.014
- Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). BatchBALD: Efficient and diverse batch acquisition for deep Bayesian active learning. arXiv:1906.08158.
- Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. arXiv:2202.10054.
- Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. Machine Learning, 51, 181–207. doi:10.1023/A:1022859003006
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. arXiv:1612.01474.
- Mustafa, B., Riquelme, C., Puigcerver, J., Pinto, A. S., Keysers, D., & Houlsby, N. (2020). Deep ensembles for low-data transfer learning. arXiv:2010.06866.
- Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., & Courville, A. (2022). The primacy bias in deep reinforcement learning. arXiv:2205.07802.
- Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. arXiv:2304.07193.
- Sener, O., & Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. arXiv:1708.00489.
- Settles, B. (2012). Active Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning. Springer. doi:10.1007/978-3-031-01560-1
- Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee. COLT, 287–294. doi:10.1145/130385.130417
- Shi, X., Cao, W., & Raschka, S. (2023). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. Pattern Analysis and Applications, 26, 941–955. doi:10.1007/s10044-023-01181-9
- Siméoni, O., et al. (2025). DINOv3. arXiv:2508.10104.
- Zaidi, S., Zela, A., Elsken, T., Holmes, C., Hutter, F., & Teh, Y. W. (2021). Neural ensemble search for uncertainty estimation and dataset shift. arXiv:2006.08573.
