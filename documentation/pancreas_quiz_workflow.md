# Pancreas subtype quiz: end-to-end workflow

Multi-task nnU-Net (3D ResEnc M) that segments pancreas/lesion and classifies the lesion subtype from
a shared encoder. This page is the runbook; the design rationale lives in the module docstrings.

## Components added to this fork

| File | Purpose |
| --- | --- |
| `nnunetv2/dataset_conversion/Dataset001_PancreasQuiz.py` | Builds the nnU-Net dataset from `train/`, `validation/`, `test/` |
| `nnunetv2/training/nnUNetTrainer/variants/network_architecture/resenc_unet_with_cls.py` | Cross-attention pooling head + multi-task wrapper |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiTaskSubtype.py` | Joint seg + subtype trainer |
| `nnunetv2/training/logging/multitask_logger.py` | Per-task curves and classification metrics to W&B |
| `nnunetv2/evaluation/quiz_metrics.py` | Metrics Reloaded suite |
| `nnunetv2/inference/predict_quiz.py` | Sliding-window seg + per-case subtype aggregation |
| `nnunetv2/evaluation/evaluate_quiz.py` | Held-out evaluation + submission packaging |

## 0. Environment

```bash
export nnUNet_raw=".../nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_data/nnUNet_preprocessed"
export nnUNet_results=".../nnUNet_data/nnUNet_results"

export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=uhn-pancreas-quiz
export nnUNet_wandb_mode=online     # 'offline' if the machine has no network
wandb login                          # once per machine
```

## 1. Dataset conversion

```bash
python nnunetv2/dataset_conversion/Dataset001_PancreasQuiz.py -i <folder with train/ validation/ test/> -o "$nnUNet_raw"
```

Produces `Dataset001_PancreasQuiz` with `imagesTr`/`labelsTr` (252 train cases), `imagesVal`/`labelsVal`
(36 held-out cases), `imagesTs` (72 test images), `dataset.json` and `subtype_labels.json`.

**The validation split is deliberately not in `imagesTr`.** nnU-Net addresses `imagesTr`, `labelsTr` and
`imagesTs` by fixed name, so `imagesVal`/`labelsVal` are invisible to fingerprinting, planning,
preprocessing and training. Putting the 36 cases in `imagesTr` behind a custom `splits_final.json`
would keep them out of the gradients but still leak their intensity and spacing statistics into the
dataset fingerprint.

## 2. Plan and preprocess (ResEnc M is mandatory for this task)

```bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity
```

Produces `nnUNetResEncUNetMPlans.json`: patch `[64, 128, 192]`, batch 2, 6 stages, features to 320.

## 3. Train

```bash
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMultiTaskSubtype
```

### Convergence detection

The brief says to stop when the loss converges rather than run the default 1000 epochs. The trainer
detects the plateau itself.

The subtlety is that nnU-Net anneals the learning rate over the *full* `num_epochs`
(`lr = initial_lr * (1 - epoch/num_epochs)**0.9`). Killing a 1000-epoch run at epoch 300 leaves the
learning rate at **72% of initial**, so the weights were never annealed and the model you keep is
worse than one that ran the schedule out. Plain early stopping on nnU-Net costs accuracy.

So detection has two stages: when the monitored metric plateaus, the learning rate is decayed from
wherever it is down to ~0 over `convergence_anneal_epochs`, and only then does training stop.

| Attribute | Default | Meaning |
| --- | --- | --- |
| `enable_early_stopping` | True | False = detect and log only, keep training |
| `convergence_metric` | `combined` | `combined` (mean of EMA pseudo-Dice and **case-level** macro-F1), `ema_fg_dice`, or `cls_macro_f1` |
| `convergence_patience` | 50 | epochs without improvement before declaring a plateau |
| `convergence_min_delta` | 1e-3 | improvement needed to count |
| `convergence_min_epochs` | 100 | never converge before this (raised to at least `cls_warmup_epochs`) |
| `convergence_smoothing` | 0.9 | EMA on the monitored value; the raw signal is noisy |
| `convergence_anneal_epochs` | 25 | length of the anneal-out phase; must be >= 2 |

`combined` is the default because segmentation converges earlier than classification here —
classification has a warmup and only 252 case labels, so stopping on `ema_fg_dice` alone would cut it
off while it is still improving.

State is written to `convergence_state.json` in the fold folder, so `--c` resumes mid-anneal and
still stops at the original epoch instead of restarting the anneal.

The most reliable option is still a fixed budget that anneals fully. Run once with
`enable_early_stopping = False`, read the reported best epoch, then set `num_epochs` near
`best_epoch + anneal_epochs` for the real run:

```python
self.num_epochs = 250
```

## 3b. Running folds in parallel, one per GPU

Two different things, do not confuse them:

| | What it does |
| --- | --- |
| `nnUNetv2_train ... -num_gpus N` | DDP: **one** fold, batch split across N GPUs |
| `nnUNetv2_train_folds_parallel` | **N folds** at once, one per GPU, each an ordinary single-GPU run |

For cross-validation you want the second — folds are independent, so running them concurrently is
statistically identical to running them sequentially, just faster in wall-clock.

```bash
nnUNetv2_train_folds_parallel -d 1 -c 3d_fullres -f 0 1 2 3 4 -g 0 1 \
    -tr nnUNetTrainerMultiTaskSubtype -p nnUNetResEncUNetMPlans --log_folder logs/
```

Folds are pulled from a shared queue, so with 5 folds on 2 GPUs a GPU that finishes early takes the
next fold rather than idling. GPU selection is done with `CUDA_VISIBLE_DEVICES` per child process,
which is what nnU-Net requires (`-device` must not be used to pick a GPU index). W&B runs are grouped
via `WANDB_RUN_GROUP`/`WANDB_NAME` so the folds appear as one experiment.

Caveats:
- run `nnUNetv2_plan_and_preprocess` **once** first; concurrent folds share the preprocessed data read-only
- each fold spawns its own augmentation workers, so CPU and RAM are the usual bottleneck before GPU is
- Colab and Kaggle normally give one GPU; this matters on a multi-GPU cluster node (Kaggle sometimes offers 2x T4)

Hyperparameters (class attributes on `nnUNetTrainerMultiTaskSubtype`):

| Attribute | Default | Meaning |
| --- | --- | --- |
| `cls_loss_weight` | 0.5 | weight of the classification CE in the joint loss |
| `cls_warmup_epochs` | 25 | linear ramp-in of that weight |
| `cls_query_num` | 4 | learned query tokens in the pooling |
| `cls_num_heads` | 4 | attention heads |
| `cls_dropout` | 0.3 | dropout in the head |
| `cls_label_smoothing` | 0.1 | label smoothing in the classification CE |
| `use_cross_attention` | True | set False for a global-average-pooling baseline |
| `cls_patch_weighting` | True | weight the classification loss by how much evidence each patch holds |
| `cls_patch_weight_floor` | 0.3 | weight of a patch with pancreas but no lesion |

### Patch weighting for the classification loss

Every training patch inherits the subtype of the case it was cropped from, but a patch that contains
no pancreas cannot possibly show that subtype — the target is pure label noise and contributes only
gradient variance. With `oversample_foreground_percent = 0.33` and batch 2, one patch per batch is
forced to contain foreground and nnU-Net picks the class uniformly, so only about half of all patches
are lesion-centred.

The loss therefore grades each patch:

| Patch content | Weight |
| --- | --- |
| no foreground at all | 0 |
| pancreas, no lesion | `cls_patch_weight_floor` |
| lesion present | 1 |

Lesion-free pancreas patches are down-weighted rather than dropped because parenchymal texture,
atrophy and duct calibre still carry subtype signal. Set `cls_patch_weighting = False` for the
unweighted ablation.

The weighted mean reproduces PyTorch's `reduction='mean'` denominator (the sum of the target classes'
CE weights, not the batch size), so `cls_loss_weight` means the same thing with weighting on or off.
`train/cls_patch_weight` in W&B tracks the mean weight per batch: if it sits near the floor, the
sampler is rarely hitting the lesion and `oversample_foreground_percent` should be raised.

### What gets reported where

| Cadence | What | Split |
| --- | --- | --- |
| every epoch | seg + cls loss, pseudo-Dice | internal fold-val split of the 252, on **patches** |
| every epoch | balanced accuracy / macro-F1 / MCC (`val_case_cls/*`) | internal fold-val split, patches aggregated **per case** |
| every epoch | macro-F1 (`val_patch_cls/*`) | same, left at **patch** level as a diagnostic |
| end of training | real sliding-window Dice | internal fold-val split, whole cases |
| on demand (step 4) | full Metrics Reloaded suite | the 36 held-out cases |

The quiz scores one subtype per case, so the per-epoch classification metrics aggregate the internal
split's patch probabilities by case before scoring — this is the training-time mirror of the
aggregation in `predict_quiz.py`, using uniform pooling (the conservative variant; the submitted
predictions use foreground-weighted pooling and score at least as well). The patch-level macro-F1 is
kept alongside it but understates real performance by construction, because it counts patches whose
label is not inferable from the patch. The case-level number is what feeds convergence detection.

`checkpoint_best` tracks the segmentation pseudo-Dice EMA, exactly as in stock nnU-Net. The 36 held-out
cases are never used for checkpoint selection — using them would bias the number you report.

## 4. Predict

```bash
MODEL="$nnUNet_results/Dataset001_PancreasQuiz/nnUNetTrainerMultiTaskSubtype__nnUNetResEncUNetMPlans__3d_fullres"

# held-out validation
nnUNetv2_predict_quiz -m "$MODEL" -i "$nnUNet_raw/Dataset001_PancreasQuiz/imagesVal" -o predictions/val -f 0

# test set
nnUNetv2_predict_quiz -m "$MODEL" -i "$nnUNet_raw/Dataset001_PancreasQuiz/imagesTs" -o predictions/test -f 0
```

Each output folder gets `quiz_*.nii.gz`, `subtype_results.csv` (`Names`, `Subtype`), plus
`subtype_probabilities.{csv,json}` which the evaluator needs for AUROC and average precision.

Cases span several sliding-window patches (94 of the 252 training cases need 1 patch, the median is 2,
the maximum 12), so per-patch subtype probabilities are averaged into one prediction per case, weighted
by how much pancreas each patch contains according to that same forward pass. `--uniform_pooling`
switches to a plain average.

## 5. Evaluate on the held-out validation split

```bash
nnUNetv2_evaluate_quiz -i predictions/val -d "$nnUNet_raw/Dataset001_PancreasQuiz" \
    -o predictions/val/summary.json --track undergraduate
```

Prints the metric table and a PASS/FAIL against the brief's thresholds (undergraduate: whole-pancreas
DSC 0.90+, lesion DSC 0.27+, macro-F1 0.60+).

## 6. Package the submission

```python
from nnunetv2.evaluation.evaluate_quiz import package_submission
package_submission('predictions/test', 'your_name_results.zip')
```

Writes the 72 test segmentations plus `subtype_results.csv` flat at the archive root.

## Tests

```bash
python -m pytest nnunetv2/tests/test_quiz_metrics.py nnunetv2/tests/test_multitask_logger.py \
                 nnunetv2/tests/test_resenc_unet_with_cls.py nnunetv2/tests/test_multitask_trainer.py \
                 nnunetv2/tests/test_quiz_inference.py -q
```

`-m "not slow"` skips the tests that build the full network and run training epochs on CPU.
