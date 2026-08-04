# Data augmentation and TTA ablations

## Bottom line

The alternate augmentation approaches were run, but none replaced the accepted configuration.
The result is nuanced:

- 8-view spatial flip TTA **did help the original cross-attention ensemble** on the provided
  held-out set and remains part of the final system.
- Training-time image augmentation did not produce a classifier that transferred better across
  folds and the provided held-out set.
- Feature-space noise/dropout gave a small OOF improvement in one MLP experiment, but the held-out
  score fell. It was rejected as non-transferring regularization rather than called a total failure.
- A stricter no-geometry PANDA high-signal trainer was implemented as a possible follow-up but was
  not run. It is not evidence and must not appear as a completed ablation.

## Training-time image augmentation

### Cue-preserving mild augmentation

The main corrective augmentation stack used:

- rotations within +/-10 degrees;
- isotropic scaling from 0.9 to 1.1;
- axis mirroring;
- Gaussian noise, brightness, and contrast transforms;
- no Gaussian blur, simulated low resolution, gamma, or intensity inversion.

This was deliberately milder than default nnU-Net geometry because lesion scale and calibrated CT
appearance may carry subtype information.

| Experiment | Evaluation | Macro-F1 | Reference | Conclusion |
|---|---|---:|---:|---|
| Continue the trained cross-attention head with mild DA, best-F1 checkpoint | five-fold OOF, 252 | 0.3982 | original head no-TTA OOF 0.3983 | Neutral |
| Same continuation, best-AUROC checkpoint | five-fold OOF, 252 | 0.4059 | original head 8-view OOF 0.4003 | Too small to justify held-out expansion |
| Fresh cross-attention head with mild DA | five-fold OOF, 252 | 0.4332 | original head no-TTA OOF 0.3983 | OOF improvement, but failed transfer |
| Fresh mild-DA head | held-out, no TTA, 36 | 0.3889 | original held-out no-TTA 0.4583 | Worse |
| Fresh mild-DA head | held-out, 8-view TTA, 36 | 0.5422 | original held-out 8-view 0.5833 | Worse |

The first row is the clean augmentation/continued-training control. Its five-fold result is
essentially unchanged. The fresh-head experiment learned a somewhat better OOF boundary but did
not beat the original model on held-out data, so the improvement was not robust.

### Restoring richer intensity transforms

A controlled fold-0 experiment restored Gaussian blur, simulated low resolution, and gamma while
keeping +/-10-degree rotations and 0.9-1.1 scaling. It scored **0.3667 fold-0 OOF macro-F1**, versus
**0.5142** for the corresponding fresh-head mild-DA fold-0 result. The richer intensity stack was
therefore rejected before a five-fold expansion.

A plausible explanation is that blur, low-resolution simulation, and gamma perturb subtle lesion
texture or calibrated CT-intensity cues more than they improve invariance. With only 252 cases,
regularization cannot compensate if the transforms partly erase the class signal.

### PANDA-style geometry tests

The from-scratch Phase-B cross-attention model inherited default 3D nnU-Net transforms: rotations
up to +/-30 degrees, scaling from 0.7 to 1.4, mirroring, noise, blur, brightness, contrast,
low-resolution simulation, and gamma. It collapsed to subtype 1 and scored **0.2009 fold-0 OOF**.

A pooling-MLP pilot reduced geometry to +/-10 degrees and 0.9-1.1 while retaining the broad
intensity stack. It improved to **0.2800 fold-0 OOF**, but still predicted almost no subtype-2
cases and failed the continuation gate. A near-whole-ROI pooling MLP using the original same-data
backbone also collapsed at **0.2009 fold-0 OOF**.

These are not clean augmentation-only comparisons because architecture, patch size, and trainable
parameters also changed. They support the broader conclusion that augmentation was not the only
problem: the classifier/representation and patch-to-case supervision remained limiting.

## Feature-space augmentation

For the encoder mean/max MLP, Gaussian feature noise (`std=0.05`) and feature dropout (`p=0.10`)
were applied after feature standardization.

| MLP setting | Five-fold OOF macro-F1 | Held-out macro-F1 | Decision |
|---|---:|---:|---|
| Aligned encoder/MLP pairs, no feature augmentation | 0.5312 | 0.5839 for the accepted MLP family | Baseline |
| Noise + feature dropout | 0.5358 | 0.5500 | Reject; small OOF gain did not transfer |
| Noise + feature dropout, second seed | 0.5383 | 0.4311 | Reject; seed instability |
| All five encoder members as training feature variants, no noise | 0.5066 | not expanded | Reject |
| All five members + noise/dropout | 0.5314 | not expanded | No advantage over aligned baseline |
| Last-mile feature-noise MLP fused with original TTA branch | 0.5386 MLP OOF | 0.6046 fused | Below accepted 0.6296 |

The OOF gains were only 0.005-0.007 and were smaller than seed variation. The held-out failures show
why augmentation was not selected based solely on its best OOF score.

## Inference-time augmentation

| TTA | Model/split | Macro-F1 | Decision |
|---|---|---:|---|
| No flips | original cross-attention, OOF | 0.3983 | Baseline |
| 8 flip views | original cross-attention, OOF | 0.4003 | Nearly neutral OOF |
| 8 flip views | original cross-attention, held-out | **0.5833** | Retained |
| Axis-0 flip pair | original cross-attention, OOF | 0.4156 | Did not establish held-out gain |
| Axis-1 flip pair | original cross-attention, OOF | 0.4157 | Held-out fell to 0.4849 |
| Axis-2 flip pair | original cross-attention, OOF | 0.3803 | Reject |
| 8 flips plus gamma 0.85/1.15 | original cross-attention, OOF | 0.3983 | Reject; no gain |
| Flip TTA | encoder mean/max MLP, held-out | 0.5241 | Reject vs 0.5839 no-TTA |

The final asymmetric inference choice is therefore intentional: 8-view flips for cross-attention,
no TTA for the compact MLP. TTA is not universally beneficial; it helps only the branch whose
learned probabilities are sufficiently equivariant and complementary after averaging.
