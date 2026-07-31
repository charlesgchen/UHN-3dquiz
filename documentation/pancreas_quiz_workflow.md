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

The brief says to stop when the loss converges rather than run the default 1000 epochs. To shorten:

```python
# in a subclass, or edit the attributes on nnUNetTrainerMultiTaskSubtype
self.num_epochs = 250
```

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

### What gets reported where

| Cadence | What | Split |
| --- | --- | --- |
| every epoch | seg + cls loss, pseudo-Dice, balanced accuracy / macro-F1 / MCC | internal fold-val split of the 252, on **patches** |
| end of training | real sliding-window Dice | internal fold-val split, whole cases |
| on demand (step 4) | full Metrics Reloaded suite | the 36 held-out cases |

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
