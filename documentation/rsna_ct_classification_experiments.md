# CT-only classification experiment record

Last updated: 2026-08-04 UTC

## Scope and non-negotiable guardrails

The README's master/PhD target is case-level three-class macro-F1 greater than 0.70. The final
operational acceptance target requested on 2026-08-03 is held-out macro-F1 at least approximately
0.62. All final candidates use only the
provided preprocessed 3D CT volumes and training annotations. Morphology, case-level tabular
features, ExtraTrees, external datasets, and externally pretrained weights are excluded.

The first experiment phase kept the already-good nnU-Net segmentation model frozen. PANDA-style
joint training from random initialization was then tested as an explicit fallback and rejected. The
current phase reuses the original five fold-matched nnU-Net checkpoints trained from random
initialization on this supplied dataset. These are not public or external pretrained weights. The
segmentation decoder is never optimized; experiments reset or replace the neural classification
head and, only in the user-approved capacity tests, adapt part or all of the original encoder.

All formal training and evaluation runs are logged online in
`charlesg-chen-university-of-toronto/uhn-pancreas-quiz`.

## Evaluation and decision protocol

Cheap hypotheses are rejected with cross-validated out-of-fold (OOF) inference before spending five
folds of GPU time. The 36-case held-out set is used only after an OOF candidate is competitive. Macro
F1 is always the primary classification metric; AUROC, balanced accuracy, per-class sensitivity,
specificity, and F1 are diagnostics. Patch-level validation scores are not reported as case-level
OOF results.

The strongest allowed historical neural held-out result before the final product ensemble was
macro-F1 0.5839. The morphology
ExtraTrees result (0.6543) is recorded only as historical context and is prohibited from the final
solution.

## Accepted CT-neural result: macro-F1 0.6296

The requested held-out threshold was reached by an equal-weight geometric ensemble of two neural
classification heads over the original five same-data nnU-Net encoders:

1. The original cross-attention subtype head, evaluated with all eight flip views.
2. A 16-unit GELU MLP over a 640-dimensional encoder representation formed by concatenating the
   spatial mean and maximum of the bottleneck features. Sliding-window embeddings are weighted by
   predicted lesion/foreground evidence and averaged; no morphology measurement is computed or
   supplied to the MLP. Five MLP CV members and five nnU-Net encoder members are averaged.

The fusion is the parameter-free product-of-experts rule
`normalize(sqrt(p_original * p_encoder_mlp))`. Both weights are fixed at 0.5; no validation label is
used to fit a weight, bias, threshold, or meta-classifier. Both branches were trained only on the 252
training scans. The accepted result is stored in
`predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout` and logged as W&B run
`mwfad9f2`.

| Metric | Value |
|---|---:|
| Held-out macro-F1 | **0.6296** |
| Balanced accuracy | 0.6222 |
| Accuracy | 0.6667 |
| Macro AUROC | 0.7291 |
| Macro average precision | 0.6134 |
| Subtype 0 / 1 / 2 F1 | 0.5000 / 0.7222 / 0.6667 |

The confusion matrix is `[[3,4,2],[0,13,2],[0,4,8]]`. This improves the strongest single neural
branch by 0.0457 absolute macro-F1 and passes the requested approximately-0.62 acceptance threshold.
The original ensemble's segmentation outputs remain unchanged and already exceed the README
requirements.

### Locked test inference and submission

The validation-selected architecture and equal-geometric rule were applied unchanged to all 72
unlabelled test scans. The original head used all eight flip views; the encoder MLP used the
non-mirrored encoder embeddings, matching its accepted held-out evaluation. Encoder-MLP test
inference is W&B run `f8dup16y`, locked fusion is run `jbe5g05b`, and the verified final submission
artifact is run `4u1a4eld`.

The flat README-compliant archive is
`/workspace/app/UHN-3dquiz/submissions/charles_chen_results.zip`, with SHA-256
`5403b303206f30898b93ba346d5265d85046a64f69cc04875a64ceed2ec0c78f`. It contains exactly 72
NIfTI segmentations and `subtype_results.csv`. Archive CRC, exact case coverage, CSV labels,
probability normalization, segmentation label range, non-empty masks, and NIfTI geometry were all
verified.

## Bounded expansion of the accepted MLP + cross-attention design

After reaching 0.6296, a final low-risk search kept the same two neural branches and tested only
pooling, MLP optimization/refitting, TTA, and parameter-free fusion. New candidates were gated on
the 252-case five-fold OOF predictions before their 36-case held-out evaluation. No candidate
improved the accepted held-out result.

| Same-design hypothesis | OOF macro-F1 | Held-out macro-F1 | Decision |
|---|---:|---:|---|
| Cross-attention top-3 window pooling instead of lesion-evidence weighting | 0.4297 | 0.5568 | Reject |
| Cross-attention uniform window pooling | 0.4003 | — | Reject at OOF gate |
| Original + top-3 + baseline MLP geometric fusion | 0.4593 | 0.5629 | Reject |
| Gamma 0.85/1.15 classification TTA added to 8x flips | 0.3983 | — | Reject at OOF gate |
| Denser sliding-window step, 0.33 instead of 0.5 | 0.4003 | — | Reject at OOF gate |
| Best regularized MLP sweep member, hidden 16 / dropout 0.3 | 0.5466 | 0.5385 | Reject |
| Best OOF MLP-fusion setting, learning rate 0.003 / arithmetic fusion | 0.4829 fusion | **0.6296** | Exact tie; retain accepted model |
| Cross-attention probabilities appended to MLP features | at most 0.5225 | — | Reject at OOF gate |
| Five MLP seeds refit on all 252 training embeddings for 64 epochs | training diagnostic 1.0000 | 0.5922 MLP / 0.6237 fused | Better MLP, weaker fusion; reject |
| Five MLP seeds refit on all 252 training embeddings for 34 epochs | training diagnostic 0.9837 | 0.5425 MLP | Reject |

The 64-epoch full-data refit demonstrates that the MLP branch still had standalone headroom from
seeing all training cases, but its perfect training diagnostic and weaker fusion show that this
headroom was mostly correlated memorization rather than new complementary signal. Likewise, top-3
pooling improved the original cross-attention head on OOF but failed to transfer to held-out. The
evidence therefore supports retaining the original lesion-evidence-weighted cross-attention branch,
the cross-validated compact MLP, and their equal geometric fusion.

## What has been ruled out

| Hypothesis | Evaluation | Macro-F1 | AUROC | Decision |
|---|---|---:|---:|---|
| Class-specific radius-5 sphere heatmaps + top-k pooling | complete 2-fold OOF | 0.3068 | 0.5551 | Reject |
| Full-lesion heatmaps + top-k pooling | complete 2-fold OOF | 0.2581 | 0.4996 | Reject |
| Flip TTA on sphere heatmaps | complete 2-fold OOF | 0.3050 | 0.5346 | Reject; delta -0.0018 |
| Soft predicted-lesion-centroid raw-CT crop MLP | complete 2-fold OOF | 0.4681 | 0.6329 | Signal, but below ceiling |
| Frozen case-embedding regularized classifier | nested five-fold OOF | 0.3080 | — | Reject |
| Frozen case-bag classifier | exact fold-0 full-volume OOF | 0.3715 | 0.5742 | Reject before five-fold expansion |
| Frozen patch-attention MIL | official five-fold OOF, 252 cases | **0.3619** | **0.5217** | Reject |

The five-fold patch export itself reproduced the original reset-head classifier at macro-F1 0.3983
and AUROC 0.5923. Patch-attention MIL therefore did not merely fail to beat the 0.5833 ceiling; it
also degraded the source representation. Its confusion matrix was `[[19,13,30],[35,38,33],
[26,22,36]]`, with per-class F1 0.2676, 0.4246, and 0.3934.

Cross-fitted logit offsets also failed to transfer: the raw-CT crop model fell from 0.4681 to 0.4430,
and the best valid sphere-heatmap model fell from 0.3068 to 0.2797. This argues against a single
class-prior correction being the primary issue.

### Invalidated early experiments

The original generic sphere-MLP training path bypassed its EMA optimizer wrapper, leaving validation
and `checkpoint_best_cls.pth` on an initialization-time head. Consequently, the early compact MLP
(reported 0.2628), memory-token model (0.2645), and related ensembles are invalid tests rather than
valid negative evidence. Corrected designs use an explicit post-optimizer EMA update and tests. They
are not being prioritized because the independent, valid frozen-feature screens above already show
that the frozen representation is the limiting factor.

## PANDA-style attempts from scratch

W&B run `4z87sjea` was a substantive single-stage fold-0 run, not a short smoke test. The first launch,
`a1711ohj`, was stopped after one epoch because single-threaded augmentation repeatedly starved the
GPU; its checkpoint directory was preserved with an `aborted_nproc0` suffix. The replacement uses
eight augmentation workers but was otherwise the identical experiment. It was rejected after epoch
25: classification training CE remained approximately 1.10, balanced accuracy was exactly 0.3333
throughout, and argmax predictions remained collapsed to one class despite segmentation EMA Dice
reaching 0.648. Transient internal macro-AUROC values as high as 0.675 did not persist. The run folder
is preserved with a `rejected_single_stage_epoch26` suffix.

- Input: provided preprocessed 3D CT patches and segmentation labels only.
- Initialization: random; the trainer rejects parameter changes before epoch zero.
- Trainable network: the complete 102,932,018-parameter ResEnc-M encoder, segmentation decoder, and
  classification head.
- Classification head: four deepest encoder scales, per-scale 1x1x1 projection to 128 channels,
  adaptive 4x4x4 pooling, factorized 3D positional plus scale embeddings, four learned query tokens,
  and two cross-attention/self-attention/MLP blocks.
- Sampling: uniform subtype sampling, with 0.67 lesion-centred and 0.33 standard crops.
- Loss: native deep-supervision segmentation loss plus uniform three-class cross entropy at weight
  0.3. Classification loss is applied only to patches containing at least 32 lesion voxels.
- Optimizer: AdamW; encoder and decoder learning rate 1e-4, head learning rate 1e-3, cosine schedule,
  300 epochs, 250 training and 50 validation steps per epoch.
- Checkpoint rule: best internal case-aggregated classification macro-F1; final acceptance still
  requires exact sliding-window case-level OOF inference. New/resumed runs also preserve a separate
  best macro-AUROC checkpoint for diagnosing useful ranking hidden by class-logit bias; AUROC does
  not replace macro-F1 as the target.

The architectural smoke run `9iq5zoy9` executed two real training epochs and full validation, proving
the complete forward/backward/checkpoint/W&B path. Its low Dice after only ten optimizer steps is
expected and is not treated as evidence for or against the model. No public MedicalNet or other
external checkpoint has been loaded by either run.

That rejected experiment started segmentation and classification jointly. The alternative-plan
document also specified a stricter two-phase schedule (100–150 epochs of classification-free
segmentation pretraining from random initialization, then 150–300 joint epochs). Existing `Stable`
checkpoints were audited and were not valid substitutes for Phase A because they had themselves been
trained with a classification head. The subsequent Phase-A run was therefore created from random
initialization; older joint weights were not relabelled as segmentation pretraining.

The strict two-phase fold-0 path completed. Phase A W&B run `ma0xa8cy` trained a
classification-free ResEnc-M for 100 epochs from random initialization and obtained mean actual
fold-validation foreground Dice 0.7015. Phase B W&B run `94y305y1` loaded only its audited same-fold
best-lesion checkpoint (SHA-256
`0ff8d7590b519434057e0eab9b5b0e0aff9812d116d383d5689fb6f852a5ffeb`) and jointly trained the full
encoder, decoder, and fresh cross-attention classifier. Convergence was detected and annealed out at
epoch 124; the final actual segmentation validation mean Dice was 0.7044.

Exact 51-case full-volume fold-0 OOF inference of Phase B's best macro-F1 checkpoint was logged as
W&B run `egzk5gcr`. It produced effectively invariant probabilities near
`[0.321, 0.352, 0.327]`, classified every case as subtype 1, and scored macro-F1 **0.2009**,
balanced accuracy 0.3333, macro-AUROC 0.4612, and confusion matrix
`[[0,13,0],[0,22,0],[0,16,0]]`. This rejects the cross-attention Phase-B classifier; its good
segmentation result does not transfer to subtype classification.

### Exact augmentation configuration

The rejected cross-attention run inherited the nnU-Net 3D training transform stack. For the actual
`[64,128,192]` plan, `dummy_2D=False`, so the exact geometry was:

- rotation probability 0.20 with angles from -30 to +30 degrees on each 3D axis;
- isotropic scaling probability 0.20 with factor 0.7 to 1.4;
- elastic-deformation probability 0;
- independent mirroring on permitted axes 0, 1, and 2.

Intensity transforms were Gaussian noise (probability 0.10, variance 0–0.1), Gaussian blur
(probability 0.20, sigma 0.5–1.0 and per-channel probability 0.5), brightness multiplication
(probability 0.15, factor 0.75–1.25), contrast (probability 0.15, factor 0.75–1.25), simulated low
resolution (probability 0.25, scale 0.5–1.0 and per-channel probability 0.5), inverted gamma
(probability 0.10, gamma 0.7–1.5 with statistics retained), and ordinary gamma (probability 0.30,
gamma 0.7–1.5 with statistics retained). Spatial transforms are shared by CT and segmentation; no
transform changes the case subtype.

The subsequent quick test was W&B run `w8hbbbpi`: the required all-encoder-level avg/max-pooling MLP
baseline, with the full encoder and decoder trainable from the same audited Phase-A checkpoint.
Only its geometry was made comparable to the cited RSNA recipe: rotation was limited to +/-10
degrees and scaling to 0.9–1.1; all intensity transforms and mirroring above remained enabled. This
tested the joint hypothesis that the transformer/global-average route diluted focal evidence and
that 0.7–1.4 scaling destroyed scale-dependent subtype cues. It failed the 40-epoch classification
gate and was rejected before a longer run.

## Original five-fold encoder/head corrective phase

This phase starts from `nnUNetTrainerSubtypeHeadAdamW` fold-matched checkpoints. They contain no
external weights and were trained only on the provided CT data. All classification variants consume
3D nnU-Net encoder features. The held-out set has 36 cases; OOF has 252 cases. Results from these two
sets must not be compared as though they were the same estimate.

| Candidate | Evaluation | Macro-F1 | AUROC | Decision |
|---|---|---:|---:|---|
| Original five-fold head | held-out, no TTA | 0.4583 | 0.6732 | Baseline |
| Original five-fold head | held-out, mirror TTA | **0.5833** | 0.6918 | Former single-head incumbent |
| Encoder mean/max MLP (16 hidden units) | five-fold OOF | 0.5309 | 0.6314 | Complementary neural head |
| Encoder mean/max MLP (16 hidden units) | held-out, no TTA | 0.5839 | 0.7464 | Strongest single neural branch |
| Original 8x-TTA + encoder MLP equal geometric ensemble | held-out | **0.6296** | 0.7291 | **Accepted** |
| Fresh cross-attention head, frozen original encoder | five-fold OOF | 0.4332 | 0.5824 | Weak OOF gain |
| Fresh cross-attention head, frozen original encoder | held-out, no TTA | 0.3889 | 0.6732 | Reject |
| Fresh cross-attention head, frozen original encoder | held-out, mirror TTA | 0.5422 | 0.7047 | Reject vs 0.5833 |
| All-level avg/max pooling MLP, frozen original encoder | five-fold OOF | 0.4477 | 0.5898 | Modest gain only |
| Fixed 80/20 fresh-head/pooling-MLP blend | five-fold OOF | 0.4520 | 0.5857 | Insufficient |
| Restored blur/low-resolution/gamma augmentation | fold-0 OOF | 0.3667 | 0.6228 | Reject |
| Fresh head + deepest encoder stage adaptation | fold-0 OOF | 0.4662 | 0.6568 | Reject vs fresh fold 0 |
| Fresh head + complete encoder adaptation | fold-0 OOF | **0.5362** | **0.7180** | Passed all-fold expansion gate |
| OOF-ranked original folds 2+1 | held-out, mirror TTA | 0.5025 | 0.7130 | Reject; lost ensemble diversity |
| Original five-fold / top-two fixed equal blend | held-out, mirror TTA | 0.5362 | 0.7200 | Reject |
| Original folds 1-4 + fresh replacement fold 0 | held-out, mirror TTA | 0.5370 | 0.7053 | Reject |

The replacement experiment directly tested whether the weakest original member should be removed.
The original folds 1-4 scored 0.5370, and adding the fresh fold-0 member with one-fifth total weight
changed no final class decisions. The original fold 0 is weak alone but contributes useful diversity
to the 0.5833 ensemble. On its OOF split, the fresh fold-0 model improved primarily subtype 2, not
subtype 0; on held-out TTA it became subtype-2 biased and its subtype-0 F1 fell to 0.1667.

The final capacity candidate was a fresh cross-attention head plus complete encoder adaptation. It
kept the entire segmentation decoder frozen and used head/encoder learning rates `1e-4`/`5e-6`,
uniform CE with subtype-balanced case sampling, mild geometry, source-weight anchoring, and EMA.
Although fold 0 scored 0.5362 exact OOF versus 0.5142 for the frozen-encoder fresh head, its complete
five-fold OOF score was only 0.4659 and its held-out score was 0.5407. It was rejected.

## Working hypotheses

1. **The frozen representation is not subtype-separable.** Multiple heads with very different
   pooling and imbalance treatments converge near chance AUROC. Joint gradients must reshape the CT
   features, not just the decision boundary.
2. **A case label attached to arbitrary patches creates label noise.** Lesion-centred sampling and a
   minimum lesion-content gate increase the fraction of patches whose class label is visually
   grounded.
3. **The subtype signal is multiscale.** Local lesion texture alone is insufficient; coarse organ
   context and finer lesion evidence must interact. Cross-attention over the four deepest encoder
   scales tests this directly.
4. **Sampler and loss reweighting should not be compounded.** Uniform subtype sampling already
   corrects exposure. Uniform CE avoids the unstable minority over-correction seen with multiple
   simultaneous imbalance mechanisms.
5. **TTA cannot rescue an uninformative representation.** Flip TTA was neutral/negative for the
   sphere model. It remains a low-risk inference follow-up only after a model has learned useful OOF
   ranking.
6. **The default geometric range may erase subtype cues.** The original 0.7–1.4 scale range is much
   broader than the cited +/-10% recipe and can alter lesion size/shape, which may be predictive for
   the three subtypes. The pooling-MLP gate retains broad intensity augmentation while testing mild
   geometry.

## W&B provenance

| Experiment | W&B run ID(s) |
|---|---|
| Sphere MLP smoke / early screen (EMA-invalid family) | `f19z23nd`; `uqcsrgam`, `i107efhy` |
| Compact sphere MLP / OOF (EMA-invalid family) | `fyixz5ay`; `h6y2ouse`, `l28h925z`; `hvrtlpzs` |
| Memory tokens / OOF (EMA-invalid family) | `1w5l9h7u`; `qk7kamiw`, `f2hfxmhs`; `gp1ug5yi` |
| Class-specific sphere heatmaps / OOF | `o8aye0mp`; `6hg921jn`, `2mn7l317`; `94j4asyv` |
| Full-lesion heatmaps / OOF | `86cq2z67`, `hvk6q95s`; `8k0928th` |
| Sphere flip-TTA | `ao29gs72` |
| Raw-CT crop MLP / OOF | `czmdga8v`, `ar5xesi7`; `pdyz22mp` |
| Raw-CT logit-offset calibration | `t8yd4q8v` |
| Frozen case-bag fold-0 train / exact OOF | `f3n2modd`; `rjwyup42` |
| Frozen case-embedding nested five-fold OOF | `8hjid265` |
| Five-fold OOF patch export | `hf7x0uxd` |
| Held-out patch export | `sxr92ivz` |
| Frozen patch-attention MIL five-fold OOF | `9dtia8lv` |
| PANDA joint architectural smoke | `9iq5zoy9` |
| PANDA joint single-stage fold 0 | `a1711ohj` (throughput-aborted after epoch 0); `4z87sjea` (rejected epoch 25) |
| PANDA strict Phase A fold 0 | `ma0xa8cy` (complete) |
| PANDA strict cross-attention Phase B fold 0 | `94y305y1` (rejected) |
| PANDA strict Phase-B best-classification exact OOF | `egzk5gcr` (rejected) |
| PANDA pooling-MLP + mild-geometry pilot fold 0 | `w8hbbbpi` (rejected) |
| Original five-fold held-out mirror-TTA baseline | `jl9wh4bc` |
| Fresh-head held-out no-TTA / mirror-TTA | `agg1ytyv`; `3do5lanc` |
| Pooling-MLP five-fold OOF / fixed fresh-pool blend | `5kh88dq6`; `y9ecoqpx` |
| Restored-intensity fold-0 OOF, best-F1 / best-AUROC checkpoints | `3yyvfd0l`; `3566xaa4` |
| Deepest-stage adaptation train / OOF checkpoints | `h1rpvzfm`; `ksa69ee1`, `dfp2ngun` |
| Original top-two held-out no-TTA / TTA / fixed blend | `p4qb87d4`; `n1i31y9v`; `96xa656n` |
| Original folds 1-4 / fresh fold 0 / exact replacement ensemble | `nueaw6vz`; `enqnl82v`; `b2a9gs6h` |
| Complete-encoder fresh-head fold-0 train / best-F1 / best-AUROC OOF | `jtym8dfp`; `suh7a9mo`; `4011o6x0` |
| Complete-encoder fresh-head folds 1-4 | `a5n7xnda`; `l4rkun3y`; `ymgd1qj3`; `418t42qg` (complete; rejected) |
| Encoder mean/max MLP five-fold OOF / held-out | `i1jb6pvz`; `yc8yrzsb` |
| Accepted original-TTA / encoder-MLP geometric ensemble | `mwfad9f2` |
| Full-encoder cross-attention case-bag train / fold-0 OOF | `bpgkxqfh`; `dyl16ppt`, `95xxvuwz` (rejected) |
| Multiscale pooling-MLP full-encoder case-bag fold 0 | `xlfumhcb` (aborted/rejected after class collapse) |
| Cross-attention top-3 OOF / held-out | `0zxukorg`; `srykgrfx` |
| Cross-attention uniform-pooling OOF | `pp87yf0q` |
| Original + top-3 + baseline-MLP triple fusion held-out | `u8ttmfia` |
| Gamma classification-TTA OOF | `qmotoli6` |
| Dense 0.33 sliding-window OOF | `o34so8qd` |
| MLP hyperparameter sweeps | group `last-mile-mlp-oof-tuning-20260803` |
| Feature-noise MLP geometric fusion held-out | `73jiphm3` |
| Learning-rate-0.003 arithmetic fusion / weight-1.5 geometric fusion held-out | `7k202yjb`; `9wy27a8t` |
| Learned cross-attention-probability MLP sweep | group `last-mile-crossfusion-oof-20260803` |
| Full-data 64-epoch MLP train / held-out / geometric fusion | `9ndu6d0k`; `mdcai4xi`; `jbvs8xpp` |
| Full-data 34-epoch MLP train / held-out | `qykurfub`; `uncmgzra` |
| Continued cross-attention head, mild-DA five-fold OOF best-F1 / best-AUROC | `u311o13g`; `y52vx4ou` |
| Paired encoder MLP no-noise / noise-dropout OOF / augmented held-out | `9ys0h0ks`; `gjz0y00g`; `oxa46pex` |
| Paired encoder MLP noise-dropout seed 20260804 OOF / held-out | `a86c2nhl`; `s14v71jq` |
| All-member encoder-feature baseline / augmented OOF | `rgjs3w90`; `uhs9slja` |
| Matched GAP folds 0-4 | `3oqg3x3t`; `kmxjk8de`; `mbe9h63y`; `b2wemjrm`; `nnx6h0y0` |
| Matched GAP OOF no-TTA / 8x-TTA | `0360xwn2`; `2xxb79xy` |
| Matched GAP held-out no-TTA / 8x-TTA | `fochbcey`; `6s0f29f4` |

## Final report-critical augmentation and pooling ablations

The remaining five-fold controls were completed after the initial record. Continuing the trained
cross-attention heads with cue-preserving mild augmentation scored 0.3982 five-fold OOF at the
best-F1 checkpoint and 0.4059 at the best-AUROC checkpoint. This is effectively unchanged from the
original head's 0.3983 no-TTA and 0.4003 8x-TTA OOF results. A fresh mild-DA head reached 0.4332
OOF but transferred poorly to held-out evaluation (0.3889 no-TTA and 0.5422 with 8x TTA, versus
0.5833 for the original TTA ensemble).

Feature noise (standard deviation 0.05) plus 0.10 feature dropout increased the aligned encoder-MLP
OOF result from 0.5312 to 0.5358, but its held-out result was only 0.5500. A second seed scored
0.5383 OOF and 0.4311 held-out. Using all five encoder members as training-time feature variants
scored 0.5066 without feature perturbation and 0.5314 with it. These small OOF changes were within
seed sensitivity and did not transfer, so feature augmentation was rejected.

Finally, a matched global-average-pooling head was trained for every fold from exactly the same
same-data checkpoints as the cross-attention control. GAP scored 0.2871/0.2887 OOF and
0.4293/0.5300 held-out without/with 8x TTA. The corresponding cross-attention scores were
0.3983/0.4003 OOF and 0.4583/0.5833 held-out. This completes the report's clean pooling ablation
and supports retaining cross-attention. The watcher verified all report-critical summaries at
2026-08-04T09:40:07Z; no further GPU experiment is required.

## Current status

The requested approximately-0.62 acceptance threshold has been achieved: the locked CT-neural
product ensemble scores held-out macro-F1 **0.6296**. Macro-F1 greater than 0.70 has not been
achieved. The accepted system contains no morphology, engineered case features, tabular classifier,
external dataset, or external pretrained weights. The bounded post-acceptance search over the same
MLP + cross-attention design is complete. Its strongest tuned fusion tied 0.6296 exactly but did not
improve it, so the locked submission and its archive hash remain unchanged. The normalized metrics,
W&B run registry, augmentation conclusions, and artifact manifest are consolidated under
`documentation/final_report/` in the `master` worktree.
