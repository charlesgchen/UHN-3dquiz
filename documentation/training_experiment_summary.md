# Training experiment summary

## Outcome

The final compliant system keeps the original five-fold nnU-Net 3D ResEnc-M segmentation ensemble
and combines two neural classification heads that consume its encoder feature maps. It reaches the
requested held-out classification threshold without morphology, engineered case features, tabular
models, external datasets, or external pretrained weights.

| Metric | Requested target | Final held-out result |
|---|---:|---:|
| Whole-pancreas DSC | 0.90+ | **0.9325** |
| Pancreas-lesion DSC | 0.27+ | **0.6308** |
| Classification macro-F1 | approximately 0.62+ | **0.6296** |

Final classification accuracy is 0.6667, balanced accuracy is 0.6222, macro AUROC is 0.7291, and
the confusion matrix is `[[3, 4, 2], [0, 13, 2], [0, 4, 8]]`.

## Final classification system

The two branches are:

1. The original five-fold cross-attention classification head with eight-view flip TTA.
2. A five-member 16-unit GELU MLP head over frozen nnU-Net encoder outputs. Its 640-dimensional
   input concatenates spatial mean and maximum bottleneck activations. Sliding-window embeddings are
   weighted by neural lesion/foreground evidence and averaged; no mask measurement or morphology
   descriptor is exposed to the MLP.

Their probabilities are fused with the parameter-free equal geometric rule
`normalize(sqrt(p_original * p_encoder_mlp))`. No validation example is used as training data, and
no validation-fitted weight, bias, threshold, or meta-classifier is present. Both branches were
trained only from the supplied 252 training CT scans and annotations.

## Main experiments

| Approach | Evaluation macro-F1 | Result |
|---|---:|---|
| Original five-fold classification head, 8x TTA | 0.5833 held-out | Strong base branch. |
| Frozen encoder mean/max MLP | 0.5309 five-fold OOF; 0.5839 held-out | Strong complementary neural head. |
| Equal arithmetic fusion of the two neural heads | 0.6086 held-out | Improvement, but below acceptance. |
| Equal geometric fusion of the two neural heads | **0.6296 held-out** | **Accepted result.** |
| Fresh frozen cross-attention head | 0.4332 five-fold OOF | Did not generalize. |
| Frozen all-level pooling MLP | 0.4477 five-fold OOF | Below the encoder mean/max MLP. |
| Complete-encoder fresh-head fine-tuning | 0.4659 five-fold OOF; 0.5407 held-out | Fine-tuning did not beat the original representation. |
| Full-encoder case-bag cross-attention | 0.3628 fold-0 OOF | Rejected; subtype 2 collapsed. |
| PANDA high-signal control without rotation/scaling | 0.1884 internal fold-0 gate | Completed 12 epochs; one-class collapse, not promoted to formal OOF. |
| Continued cross-attention head with cue-preserving mild augmentation | 0.3982 five-fold OOF | Effectively unchanged from the original OOF baseline. |
| Encoder-MLP feature noise/dropout | 0.5358 five-fold OOF; 0.5500 held-out | Small OOF gain did not transfer. |
| Matched global-average-pooling head, 8x TTA | 0.2887 five-fold OOF; 0.5300 held-out | Clean ablation supports cross-attention. |
| Partial-axis and full 8x TTA variants | Up to 0.5833 held-out | Helpful only for the original head. |
| Class-bias calibration, fold pruning/replacement, and probability blends | Below 0.5833 held-out | Rejected. |
| Morphology/ExtraTrees | 0.6543 held-out | Historical diagnostic only; prohibited and excluded from the final system. |

## Final same-design tuning

The final tuning pass expanded the accepted MLP + cross-attention system without introducing a new
model family. Top-3 cross-attention window pooling improved five-fold OOF macro-F1 from 0.4003 to
0.4297, but fell to 0.5568 held-out. Gamma intensity TTA and denser sliding-window inference were
rejected at the OOF gate. Regularization, learning-rate, weight-decay, batch-size, feature-noise,
class-weight, and direct neural-probability-fusion MLP sweeps also failed to transfer.

Refitting five identical compact MLPs on all 252 training embeddings produced the strongest new
standalone MLP at 0.5922 held-out, but its geometric fusion reached only 0.6237. A separate tuned
MLP with arithmetic fusion reproduced the accepted 0.6296 confusion matrix exactly rather than
improving it. The original equal-geometric system therefore remains the selected result; the locked
test predictions and submission archive were not replaced.

## Why the final fusion helps

The cross-attention head preserves localized spatial evidence but is unstable on ambiguous subtype-1
and subtype-2 cases. The compact mean/max MLP supplies a differently regularized view of the same 3D
encoder features. Geometric averaging rewards agreement and reduces an overconfident error from
either branch. It changes the held-out confusion matrix from the original head's
`[[3,4,2],[0,12,3],[0,5,7]]` to `[[3,4,2],[0,13,2],[0,4,8]]`, correcting one subtype-1 and one
subtype-2 case without sacrificing subtype 0.

## Reproducibility and artifacts

- Accepted validation output:
  `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout`
- Required classification CSV:
  `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout/subtype_results.csv`
- Original five-fold 8x-TTA branch: W&B run `jl9wh4bc`
- Encoder MLP OOF training / held-out evaluation: W&B runs `i1jb6pvz` / `yc8yrzsb`
- Accepted neural fusion: W&B run `mwfad9f2`
- Final cross-attention pooling OOF / held-out screen: W&B runs `0zxukorg` / `srykgrfx`
- Final MLP tuning: W&B group `last-mile-mlp-oof-tuning-20260803`
- Full-data MLP refit / held-out / fusion: W&B runs `9ndu6d0k` / `mdcai4xi` / `jbvs8xpp`
- Encoder MLP test inference: W&B run `f8dup16y`
- Locked equal-geometric test inference: W&B run `jbe5g05b`
- Verified final submission and provenance: W&B run `4u1a4eld`
- Final test output:
  `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_test`
- Submission archive:
  `submissions/charles_chen_results.zip`
- Submission SHA-256:
  `5403b303206f30898b93ba346d5265d85046a64f69cc04875a64ceed2ec0c78f`
- Report-ready index, normalized metrics, W&B registry, and artifact manifest:
  `documentation/final_report/`
- Detailed CT-only experiment record:
  `documentation/rsna_ct_classification_experiments.md`

All formal training and evaluation runs are recorded in the W&B project
`charlesg-chen-university-of-toronto/uhn-pancreas-quiz`.

The final archive contains exactly 72 root-level NIfTI segmentations plus one
`subtype_results.csv`. Its CRC, unique-member count, exact test-case coverage, CSV schema and class
range, probability normalization, NIfTI label range, non-empty predictions, and source-image
geometry were all checked before packaging.
