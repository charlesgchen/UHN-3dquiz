# Winning CT-neural solution

Last verified: 2026-08-05 UTC

## Executive result

The accepted system is an equal geometric fusion of two classifiers that consume the deepest
features of the original five-fold 3D ResEnc-M nnU-Net. The first is the original cross-attention
head with 8-view flip test-time augmentation (TTA); the second is a compact mean/max-pooled encoder
MLP without TTA. The original segmentation ensemble is retained unchanged.

On the 36-case provided held-out validation split, the system achieves **0.6296 macro-F1** and
**0.4905 Matthews correlation coefficient (MCC)**. It passes every undergraduate requirement:

| Required endpoint | Held-out result | Requirement | Absolute margin | Status |
|---|---:|---:|---:|---|
| Whole-pancreas DSC | **0.9325** | 0.90 | +0.0325 | Pass |
| Pancreas-lesion DSC | **0.6308** | 0.27 | +0.3608 | Pass |
| Three-class macro-F1 | **0.6296** | 0.60 | +0.0296 | Pass |

MCC is the Metrics Reloaded primary classification summary for this imbalanced multiclass problem;
macro-F1 remains the official acceptance endpoint specified by the quiz. MCC has no separate quiz
threshold.

## Scope and compliance

The final system uses only the supplied processed 3D CT volumes and labels. Every reused nnU-Net
checkpoint was initialized randomly and trained on the supplied 252-case training set. Reusing
these same-data checkpoints is not external pretraining.

The final model does **not** use morphology, lesion volume, shape, connected-component counts,
centroids or coordinates, clinical variables, case-level tabular features, ExtraTrees, external
data, or public pretrained weights. Predicted foreground probabilities are used only to weight
neural patch representations produced by the same CT forward pass; no measured mask statistic is
given to either classifier.

## Data and evaluation protocol

| Split | Cases | Class support (0 / 1 / 2) | Role |
|---|---:|---:|---|
| Training | 252 | 62 / 106 / 84 | Five-fold model fitting and OOF ablations |
| Held-out validation | 36 | 9 / 15 / 12 | Final five-member ensemble evaluation |
| Test | 72 | unavailable | Locked submission inference; labels unavailable locally |

For five-fold out-of-fold (OOF) evaluation, each training case is predicted only from an nnU-Net
member whose training fold excluded that case. OOF results are used for controlled comparisons.
For the held-out result, all five members vote, which measures the deployed ensemble but is not
directly comparable to single-member OOF prediction.

The 36 held-out cases were excluded from gradient fitting and checkpoint selection inside each
training run. They were, however, evaluated repeatedly during model development. The final held-out
number must therefore be described as a development-set result with possible model-selection
optimism, not as an independent estimate of deployment performance.

For context, the two accepted components have the following 252-case OOF results. These are useful
development estimates, but they represent one fold member per case rather than the five-member
ensemble used on held-out data:

| OOF classifier (252 cases) | Macro-F1 | MCC | Balanced accuracy | Macro AUROC | Macro AP |
|---|---:|---:|---:|---:|---:|
| Cross-attention, 8-view TTA | 0.4003 | 0.1174 | 0.4037 | 0.6067 | 0.4261 |
| Encoder mean/max MLP, no TTA | **0.5309** | **0.2970** | **0.5323** | **0.6314** | **0.4662** |

A directly comparable OOF prediction set was not saved for the accepted two-branch fusion, so no
fused OOF number is claimed or reconstructed after model selection.

## Shared 3D nnU-Net representation

The common backbone is the plans-driven six-stage 3D ResEnc-M nnU-Net with a
`64 x 128 x 192` patch, CT normalization, and encoder widths
`32, 64, 128, 256, 320, 320`. Its deepest feature map therefore has 320 channels. The original
five segmentation members were kept fixed during the successful classification-head experiments;
no new backbone or segmentation decoder was trained for the accepted system.

Segmentation output uses the ordinary five-fold nnU-Net ensemble and all permitted spatial mirror
views. Labels are background `0`, normal pancreas `1`, and lesion `2`.

## Classification branch A: cross-attention

The original classifier replaces global pooling with four learned queries that cross-attend over
the flattened spatial positions of the 320-channel bottleneck using four attention heads. The four
attended vectors are normalized, concatenated, and projected to three subtype logits. This preserves
localized evidence that global average pooling can dilute when the lesion occupies only a small
fraction of the CT volume.

For each fold, the classification head was reset and trained while the fold-matched encoder and
decoder remained frozen. The accepted head used inverse-frequency weighted cross-entropy, label
smoothing `0.05`, dropout `0.3`, AdamW with learning rate `1e-4` and weight decay `1e-4`, and the
best internal case-level macro-F1 checkpoint. Lesion-bearing patches receive full classification
weight, pancreas-only patches weight `0.3`, and background-only patches no classification weight.

At inference, sliding-window probabilities are pooled within each fold using predicted neural
foreground evidence. Probabilities are then averaged across the five folds and all eight
combinations of flips along the three spatial axes. This branch reaches **0.5833 held-out
macro-F1**. Gamma TTA and denser sliding windows did not improve OOF performance and are excluded.

## Classification branch B: compact encoder MLP

For every sliding-window patch, spatial mean and maximum pooling are applied to the same
320-channel bottleneck feature map. Their concatenation produces a 640-dimensional representation.
Patch representations are averaged with the same predicted foreground evidence, giving one neural
case embedding per encoder member.

The accepted classifier is a `640 -> 16 -> 3` GELU MLP with 10,307 trainable parameters. It uses
training-fold feature standardization, dropout `0.5`, inverse-frequency weighted cross-entropy,
label smoothing `0.05`, and AdamW with learning rate `1e-3` and weight decay `1e-2`. Five MLPs are
fit by stratified cross-validation on the 252 OOF encoder embeddings, so an MLP validation case is
excluded from that MLP and its input embedding also comes from an nnU-Net member that excluded it.

For a held-out case, five MLPs are evaluated on embeddings from five encoders and all 25 probability
vectors are averaged. This branch is used **without** flip TTA because TTA reduced held-out
macro-F1 from **0.5839** to **0.5241**. The accepted MLP also excludes feature noise and feature
dropout because their small OOF gains did not transfer to held-out data.

## Parameter-free fusion

Let `p_attn` and `p_mlp` be the three-class probability vectors from the two branches. The accepted
probabilities are

```text
p_final[c] = sqrt(p_attn[c] * p_mlp[c])
p_final = p_final / sum(p_final)
prediction = argmax(p_final)
```

This equal geometric product-of-experts rule was fixed without fitting a validation-set weight,
bias, threshold, or meta-classifier. It rewards agreement and tempers a confident error made by
either branch. Fusion improves all three hard-decision summaries, although the standalone MLP has
the best threshold-free ranking metrics:

| Held-out classifier (36 cases) | Macro-F1 | MCC | Balanced accuracy | Macro AUROC | Macro AP |
|---|---:|---:|---:|---:|---:|
| Cross-attention, 8-view TTA | 0.5833 | 0.3971 | 0.5722 | 0.6918 | 0.5693 |
| Encoder mean/max MLP, no TTA | 0.5839 | 0.4428 | 0.5889 | **0.7464** | **0.6506** |
| Equal geometric fusion | **0.6296** | **0.4905** | **0.6222** | 0.7291 | 0.6134 |

## Metrics Reloaded evaluation

### Segmentation

Segmentation metrics are computed separately for each held-out case and then macro-averaged; voxels
are not pooled across cases. The reported standard deviation is the population standard deviation
over the 36 per-case values.

| Region/metric | Definition | Mean | SD | Cases |
|---|---|---:|---:|---:|
| Whole-pancreas DSC | prediction/reference label `> 0` | **0.9325** | 0.0350 | 36 |
| Lesion DSC | label `== 2` | **0.6308** | 0.3125 | 36 |
| Normal-pancreas DSC | label `== 1` | 0.9052 | 0.0549 | 36 |
| Lesion F2 | label `== 2`, sensitivity weighted 4x | 0.6244 | 0.3150 | 36 |

DSC is reported because it is the quiz endpoint. Lesion F2 is included as the sensitivity-oriented
companion metric for a small target where a complete miss is clinically and technically important.

### Classification summary

| Metric | Held-out value | Reporting role |
|---|---:|---|
| Matthews correlation coefficient | **0.4905** | Metrics Reloaded primary summary |
| Macro-F1 | **0.6296** | Quiz-mandated acceptance metric |
| Balanced accuracy | 0.6222 | Macro-averaged sensitivity |
| Accuracy | 0.6667 | Diagnostic only under class imbalance |
| Macro one-vs-rest AUROC | 0.7291 | Threshold-free ranking |
| Macro average precision | 0.6134 | Ranking metric sensitive to class prevalence |

MCC uses the full multiclass confusion matrix and ranges from `-1` to `1`, with `0` indicating no
association between predictions and targets. It is more informative here than accuracy: an
always-subtype-1 majority classifier would obtain 0.4167 accuracy but only 0 MCC, 0.3333 balanced
accuracy, and 0.1961 macro-F1. Expected cost is not reported because the quiz supplies no clinical
misclassification-cost matrix; inventing one would make the result arbitrary.

### Per-class classification

Sensitivity, specificity, AUROC, and average precision are one-vs-rest. Support is the number of
held-out reference cases for that subtype.

| Subtype | Support | Sensitivity | Specificity | F1 | AUROC | Average precision |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 0.3333 | 1.0000 | 0.5000 | 0.7366 | 0.5964 |
| 1 | 15 | 0.8667 | 0.6190 | 0.7222 | 0.7492 | 0.6271 |
| 2 | 12 | 0.6667 | 0.8333 | 0.6667 | 0.7014 | 0.6166 |

Specificity is retained for completeness, but each one-vs-rest negative set is substantially larger
than its positive set. Sensitivity and per-class F1 are more discriminative diagnostics here.

The confusion matrix has reference subtype in rows and predicted subtype in columns:

```text
[[ 3,  4, 2],
 [ 0, 13, 2],
 [ 0,  4, 8]]
```

The model classifies 24 of 36 cases correctly. Its main remaining weakness is subtype 0 recall:
only 3 of 9 subtype-0 cases are recovered, while no subtype-1 or subtype-2 case is incorrectly
called subtype 0. Subtype 1 is the strongest class by F1, and four subtype-2 cases are confused with
subtype 1.

## Report-critical ablations

### Matched pooling control

| Pooling head | OOF, no TTA | OOF, 8-view TTA | Held-out, no TTA | Held-out, 8-view TTA |
|---|---:|---:|---:|---:|
| Cross-attention | **0.3983** | **0.4003** | **0.4583** | **0.5833** |
| Global average pooling | 0.2871 | 0.2887 | 0.4293 | 0.5300 |

The GAP control uses the same fold-matched source checkpoints, frozen encoder/decoder, optimizer,
loss, augmentation, checkpoint rule, and inference protocol. Only the classification pooling/head
changes. All five GAP folds completed and were logged, making this the cleanest architecture
ablation supporting cross-attention.

### Augmentation and TTA decisions

The accepted cross-attention head retains its original nnU-Net image augmentation and 8-view flip
TTA. A cue-preserving mild training stack improved a fresh head's OOF macro-F1 but reduced its
held-out 8-view result from 0.5833 to 0.5422. In the matched paired-encoder MLP control, Gaussian
feature noise and feature dropout raised OOF macro-F1 from 0.5312 to 0.5358 but produced only 0.5500
held-out macro-F1; a second seed fell to 0.4311. These non-transferring changes were rejected.

The full augmentation matrix, including the completed no-rotation/no-scaling PANDA gate, is kept in
[augmentation_ablation.md](augmentation_ablation.md) so the winning-solution narrative remains
readable without hiding negative results.

## Interpretation and limitations

The improvement is a modest, evidence-backed extension of the existing representation rather than
an architecture redesign. Cross-attention retains spatially localized evidence, while mean/max
pooling supplies a low-capacity global summary; their errors are complementary enough that fixed
geometric fusion improves macro-F1 by 0.0457 over the strongest standalone branch.

The result should still be reported conservatively:

- the held-out set has only 36 cases, including only nine subtype-0 cases, so individual errors move
  macro metrics materially;
- repeated held-out evaluation introduces selection optimism even though no held-out label entered
  gradient fitting or checkpoint selection;
- the accepted fusion has a held-out headline but no directly comparable fused five-fold OOF score;
  component OOF and ensemble held-out numbers must not be presented as the same evaluation design;
- the weak subtype-0 sensitivity is unresolved and is the clearest generalization risk;
- the 72 test labels are unavailable, so no local test macro-F1, MCC, or DSC can be claimed.

Runtime is not an undergraduate acceptance endpoint and was not used to select the model. The report
should prioritize the required metrics, controlled ablations, and limitations instead of claiming an
unmeasured inference-time improvement.

## Reproducibility and provenance

The metric implementation is [quiz_metrics.py](../../nnunetv2/evaluation/quiz_metrics.py), and the
held-out evaluator and threshold checks are in
[evaluate_quiz.py](../../nnunetv2/evaluation/evaluate_quiz.py). Exact source summaries are retained
in [data/source_summaries](data/source_summaries/README.md); the accepted classification summary is
[accepted_heldout.json](data/source_summaries/accepted_heldout.json), while the unchanged full
segmentation summary is in
[cross_attention_heldout_8x_tta.json](data/source_summaries/cross_attention_heldout_8x_tta.json).

Training, final evaluation, fusion, and artifact verification are recorded in the W&B project
`charlesg-chen-university-of-toronto/uhn-pancreas-quiz`. The winning-system provenance chain is:

| Stage | W&B run ID | Role |
|---|---|---|
| Source nnU-Net folds 0-4 | `x0pcluox`, `fffvqm32`, `22ic3abr`, `jqylrdqi`, `fm4cg0i0` | Random-init backbone/segmentation training curves |
| Cross-attention head folds 0-4 | `hyt8k0qs`, `sjzld1g6`, `6hxhbmyl`, `7a8g6y3y`, `6r7k05q7` | Frozen-backbone head training curves |
| OOF encoder embedding export | `na3sbatb` | Leakage-safe MLP training input |
| Held-out encoder embedding export | `a4liwh6e` | MLP held-out input |
| Original cross-attention held-out, 8-view TTA | `jl9wh4bc` | Accepted component evaluation |
| Encoder MLP five-fold OOF training | `i1jb6pvz` | MLP training curves and OOF metrics |
| Encoder MLP held-out evaluation | `yc8yrzsb` | Accepted component evaluation |
| Equal geometric fusion, held-out | `mwfad9f2` | Headline classification result |
| Encoder MLP test inference | `f8dup16y` | Locked test component |
| Equal geometric fusion, test | `jbe5g05b` | Locked test probabilities and labels |
| Submission verification | `4u1a4eld` | Archive integrity and provenance |

The complete indexed run list is [wandb_runs.csv](data/wandb_runs.csv), normalized experiment
metrics are in [experiment_metrics.csv](data/experiment_metrics.csv), and artifact identities are in
[artifact_manifest.csv](data/artifact_manifest.csv).

The locked test output is
`predictions/original_tta_frozen_encoder_mlp16_equal_geometric_test/`. The final submission archive
is `submissions/charles_chen_results.zip`, contains 72 NIfTI segmentations plus
`subtype_results.csv`, and has SHA-256
`5403b303206f30898b93ba346d5265d85046a64f69cc04875a64ceed2ec0c78f`. Provenance verifies what
was submitted and that it matches the locked predictions; it is not presented as additional
performance evidence.
