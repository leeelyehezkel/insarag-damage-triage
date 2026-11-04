# Licences

| Part | Licence | Notes |
|---|---|---|
| Code in this repository | PolyForm Noncommercial 1.0.0 (`LICENSE`) | non-commercial use only |
| Grading model weights | DINOv3 License (`licenses/DINOv3-LICENSE.md`) | must be shipped with the weights; publications must acknowledge DINOv3 |
| SAM 3 (building detection) | SAM License (Meta) | downloaded from Hugging Face by the user; publications must acknowledge SAM 3 |
| Labels and crops | TODO | see below |

## Open points

- **Weights and non-commercial use.** The DINOv3 License says that derivative works, which include fine-tuned weights, may be distributed only under its own terms. That licence does not forbid commercial use, so a non-commercial condition of our own on the weights may not hold. We ask for citation, but can't legally require it for the weights.
- **Dataset.** The photos and masks are published on Kaggle by the author (turkey-gt-images, turkey-gt-masks), so the author sets their licence; CC BY-NC 4.0 fits the aim of attribution and no commercial use. The rights of the photographers matter too: if the photos were collected from news or social media, they may not be the author's to license, and the licence should then cover only the labels.
