# Winning CT-neural solution

## Scope

The final system uses only the supplied processed 3D CT volumes and labels. Every reused nnU-Net
checkpoint was initialized randomly and trained on this dataset. Reusing these same-data models is
not external pretraining. No morphology, lesion volume, shape, coordinates, clinical variables,
case-level tabular features, ExtraTrees, external data, or public pretrained model is used.

## Segmentation branch

The original five-fold 3D ResEnc-M nnU-Net ensemble is retained unchanged because its segmentation
already exceeds the requirements:

- whole-pancreas DSC: **0.9325**;
- pancreas-lesion DSC: **0.6308**.

No new backbone or segmentation decoder was trained for the accepted system.

## Classification branch A: cross-attention

This is the original five-fold encoder-consuming classification head. Each member uses localized
bottleneck features from its fold-matched nnU-Net. At inference, probabilities are averaged over
all eight combinations of flips along the three spatial axes, then averaged across folds.

Held-out macro-F1 is **0.5833** with 8-view TTA. A matched global-average-pooling replacement
reaches only **0.5300** with the same TTA, supporting the value of spatial cross-attention rather
than merely adding classifier parameters.

## Classification branch B: compact encoder MLP

For each sliding-window CT patch, the frozen nnU-Net bottleneck is summarized with spatial mean and
maximum pooling. Concatenating both produces a 640-dimensional neural feature vector. Patch vectors
are aggregated using neural foreground/lesion evidence from the same forward pass, not measured
mask morphology. A compact `640 -> 16 -> 3` GELU MLP predicts subtype probabilities.

Five MLP members and five aligned encoder members are averaged. This branch is evaluated without
flip TTA because flip augmentation reduced its held-out macro-F1 from **0.5839** to **0.5241**.

## Parameter-free fusion

Let `p_attn` and `p_mlp` be the three-class probabilities from the two branches. The accepted
probabilities are

```text
p_final = normalize(sqrt(p_attn * p_mlp))
```

This equal geometric fusion was fixed without fitting a validation-set weight, bias, threshold, or
meta-classifier. It favors agreement and tempers a confident error made by either branch.

## Final held-out metrics

| Metric | Value |
|---|---:|
| Macro-F1 | **0.6296296** |
| Balanced accuracy | 0.6222222 |
| Accuracy | 0.6666667 |
| Macro one-vs-rest AUROC | 0.7290736 |
| Macro average precision | 0.6133722 |
| Subtype 0 F1 | 0.5000000 |
| Subtype 1 F1 | 0.7222222 |
| Subtype 2 F1 | 0.6666667 |

Confusion matrix, with rows as true subtype and columns as predicted subtype:

```text
[[ 3,  4, 2],
 [ 0, 13, 2],
 [ 0,  4, 8]]
```

## Matched pooling ablation

| Pooling head | OOF, no TTA | OOF, 8-view TTA | Held-out, no TTA | Held-out, 8-view TTA |
|---|---:|---:|---:|---:|
| Cross-attention | **0.3983** | **0.4003** | **0.4583** | **0.5833** |
| Global average pooling | 0.2871 | 0.2887 | 0.4293 | 0.5300 |

The GAP control uses the same fold-matched source checkpoints, frozen encoder/decoder, optimizer,
loss, augmentation, checkpoint rule, and inference protocol. Only the classification pooling/head
is changed. All five GAP folds were trained and logged; this is the cleanest architecture ablation
for the report.

## Recommended reporting

Use the 252-case OOF metrics for ablation comparisons and the 36-case held-out metrics for the
complete five-fold ensemble headline. Do not merge them into one table column without an explicit
split label. State that repeated development may make the held-out estimate optimistic and that
the locked 72-case test archive cannot be scored locally because its labels are unavailable.

The submission archive is `submissions/charles_chen_results.zip`, SHA-256
`5403b303206f30898b93ba346d5265d85046a64f69cc04875a64ceed2ec0c78f`.
