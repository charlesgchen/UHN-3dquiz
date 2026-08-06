# Complementary Local and Global Neural Evidence for Joint Pancreas Segmentation and CT Subtype Classification

This repository contains the code and experiment record for a 3D CT system that jointly segments the pancreas and pancreatic lesion and predicts one of three lesion subtypes. It is a task-specific fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), organized according to the [MICCAI reproducibility code checklist](MICCAI-Code-Checklist.md).

The submitted system combines a five-fold 3D ResEnc-M nnU-Net cross-attention classifier with eight-view flip test-time augmentation and a compact five-fold MLP over spatial mean/max-pooled bottleneck features. Their probabilities are combined by the fixed geometric rule `normalize(sqrt(p_cross_attention * p_encoder_mlp))`. No external data, public pretrained weights, morphology features, clinical variables, or tabular classifier are used.

## Results

These results are from the provided 36-case held-out development set. It was excluded from training and checkpoint selection, but was evaluated during model development, so it is not an independent deployment test set.

| Endpoint | Result | Undergraduate target | Status |
| --- | ---: | ---: | :---: |
| Whole-pancreas Dice | **0.9325** | 0.90 | Pass |
| Pancreas-lesion Dice | **0.6308** | 0.27 | Pass |
| Three-class macro-F1 | **0.6296** | 0.60 | Pass |
| Classification MCC | **0.4905** | Not thresholded | Reported |

The test set contains 72 cases and has no locally available labels. Its archive was validated for exact case coverage, CSV schema, NIfTI geometry, label range, non-empty masks, duplicate members, and ZIP CRC rather than assigned a test score.

See the [winning solution](documentation/final_report/winning_solution.md), [augmentation ablations](documentation/final_report/augmentation_ablation.md), and [report-ready evidence index](documentation/final_report/README.md) for complete results and provenance.

## Repository layout

| Path | Purpose |
| --- | --- |
| `nnunetv2/dataset_conversion/Dataset001_PancreasQuiz.py` | Convert the supplied data without validation leakage |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiTaskSubtype.py` | Joint segmentation/subtype trainer |
| `nnunetv2/training/nnUNetTrainer/variants/network_architecture/resenc_unet_with_cls.py` | Cross-attention subtype head |
| `nnunetv2/inference/predict_quiz.py` | Sliding-window and case-level inference |
| `nnunetv2/evaluation/quiz_metrics.py` | Metrics Reloaded-aligned metrics |
| `nnunetv2/evaluation/evaluate_quiz.py` | Evaluation and submission packaging |
| `documentation/` | Workflow, experiments, ablations, and evidence |
| `report_submission/` | LaTeX paper and submission material |

The full `nnunetv2/` package is retained because the custom trainer, preprocessing, plans, inference, and checkpoint format depend on those vendored framework internals.

## Environment and requirements

Reference experiments used:

| Component | Reference environment |
| --- | --- |
| OS | Linux 6.6 under WSL2, glibc 2.39 |
| CPU | 8 physical / 16 logical cores |
| RAM | approximately 50 GB |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| CUDA | driver-reported 13.3; PyTorch CUDA 13.0 build |
| Python | 3.12.3; package requires 3.10 or newer |
| PyTorch | 2.11.0+cu130 in recorded runs |
| nnU-Net fork | 2.8.1 |

A CUDA GPU is strongly recommended. Install the PyTorch build appropriate for your platform first, then install this repository:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for your platform: https://pytorch.org/get-started/locally/
python -m pip install -e .
~~~

The accepted MLP scripts record their configuration with Weights & Biases. Log in for online tracking, or set `WANDB_MODE=disabled` to run locally without uploading:

~~~bash
wandb login
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=uhn-pancreas-quiz
export nnUNet_wandb_mode=online
# export WANDB_MODE=disabled
~~~

Dependencies and console entry points are declared in [pyproject.toml](pyproject.toml). Exact packages from completed runs remain alongside local W&B metadata when those ignored experiment folders are available.

## Dataset

The dataset was supplied for the UHN pancreas 3D quiz and is not redistributed here. Obtain it from the task organizer/course distribution under its original terms.

~~~text
<DATA_ROOT>/
├── train/
│   ├── subtype0/
│   ├── subtype1/
│   └── subtype2/
├── validation/
│   ├── subtype0/
│   ├── subtype1/
│   └── subtype2/
└── test/
~~~

Each labelled subtype folder contains paired NIfTI files:

~~~text
quiz_<subtype>_<id>_0000.nii.gz   # CT image
quiz_<subtype>_<id>.nii.gz        # segmentation
~~~

The test folder contains only `*_0000.nii.gz` images. Segmentation labels are background `0`, normal pancreas `1`, and lesion `2`. Split sizes are 252 training, 36 held-out validation, and 72 test cases; training subtype support is 62/106/84 for classes 0/1/2.

## Data conversion and preprocessing

Set the nnU-Net paths to absolute locations:

~~~bash
export nnUNet_raw="/absolute/path/to/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/absolute/path/to/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/absolute/path/to/nnUNet_data/nnUNet_results"
~~~

Convert the supplied layout:

~~~bash
python nnunetv2/dataset_conversion/Dataset001_PancreasQuiz.py -i /absolute/path/to/DATA_ROOT -o "$nnUNet_raw" -np 8
~~~

This produces `imagesTr/labelsTr` for 252 training cases, `imagesVal/labelsVal` for 36 held-out cases, `imagesTs` for 72 test cases, `dataset.json`, and `subtype_labels.json`.

The held-out cases deliberately use names that nnU-Net does not inspect during fingerprinting or training, preventing leakage of their spacing and intensity statistics. The converter also rounds supplied floating segmentation values to exact integer labels while preserving geometry.

Plan and preprocess:

~~~bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity
~~~

The reference plan uses CT foreground-statistic normalization, resamples to `[2.0, 0.73046875, 0.73046875]` mm spacing, and uses a `[64, 128, 192]` voxel patch with batch size 2. Image interpolation is third order and label interpolation is first order with discrete-label handling. Cropping and padding are plans-driven.

## Training

Train one fold:

~~~bash
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMultiTaskSubtype
~~~

Train all folds concurrently on two GPUs:

~~~bash
nnUNetv2_train_folds_parallel -d 1 -c 3d_fullres -f 0 1 2 3 4 -g 0 1 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerMultiTaskSubtype --log_folder run_logs/multitask
~~~

The backbone is a six-stage 3D residual-encoder U-Net with widths `[32, 64, 128, 256, 320, 320]`. The committed subtype branch uses four learned queries, four attention heads, dropout 0.3, inverse-frequency cross-entropy, label smoothing 0.1, and classification-loss weight 0.5 warmed over 25 epochs. Lesion, pancreas-only, and background-only patches receive classification weights 1.0, 0.3, and 0.0.

The trainer detects convergence after at least 100 epochs with patience 50, then anneals the learning rate to zero over 25 epochs. Full defaults and the reason for annealed stopping are in the [workflow](documentation/pancreas_quiz_workflow.md).

Standard 3D training uses rotation/scaling, mirroring, noise/blur, brightness, contrast, simulated low resolution, and gamma transforms. Cue-preserving mild and feature-space augmentation did not improve the final result; exact settings are in [augmentation_ablation.md](documentation/final_report/augmentation_ablation.md).

## Trained models and reproducibility scope

Weights are not published as a separate download. Training writes model folders beneath `$nnUNet_results/Dataset001_PancreasQuiz/`.

When the original local artifact tree accompanies this checkout, accepted nnU-Net checkpoints are under `nnUNet_data/nnUNet_results/`, compact MLP weights are under `predictions/`, and their roles and hashes are in [artifact_manifest.csv](documentation/final_report/data/artifact_manifest.csv). These large paths are intentionally ignored by Git.

The repository includes the complete accepted post-hoc pipeline: OOF feature export, compact MLP training, held-out/test MLP inference, and arithmetic or geometric neural-probability fusion. A fresh clone still needs the supplied private dataset and trained nnU-Net checkpoints, because neither can be redistributed here.

## Accepted encoder-MLP and geometric fusion

Run these commands from the repository root. They reproduce the accepted `640 -> 16 -> 3` GELU MLP and parameter-free fusion recorded by W&B runs `i1jb6pvz`, `yc8yrzsb`, and `mwfad9f2`. Existing output directories are rejected intentionally to prevent mixed runs.

### 1. Export one OOF mean/max embedding per training case

~~~bash
python scripts/evaluate_classifier_internal_oof.py \
  --trainer nnUNetTrainerLesionCenteredSubtypeHeadAdamW \
  --folds 0 1 2 3 4 \
  --checkpoint checkpoint_best_cls.pth \
  --save-classification-embeddings \
  --output predictions/reset_head_5fold_oof_frozen_embeddings_no_tta \
  --run-name reset-head-5fold-oof-frozen-embeddings-no-tta \
  --wandb-group uhn-pancreas-frozen-embedding-mlp-20260802
~~~

For every patch, inference concatenates the spatial mean and maximum of the 320-channel bottleneck. Predicted neural foreground evidence pools those 640-dimensional patch vectors into one case vector. Every training case is exported only by the nnU-Net fold that excluded it.

### 2. Train the accepted five-fold compact MLP

~~~bash
python scripts/train_frozen_embedding_mlp.py \
  --embeddings predictions/reset_head_5fold_oof_frozen_embeddings_no_tta \
  --output predictions/frozen_embedding_mlp16_5fold_oof \
  --run-name frozen-embedding-mlp16-5fold-oof \
  --wandb-group uhn-pancreas-frozen-embedding-mlp-20260802 \
  --hidden-dim 16 \
  --dropout 0.5
~~~

This exact invocation uses stratified five-fold CV, seed `20260802`, AdamW with learning rate `1e-3` and weight decay `1e-2`, inverse-frequency class weights, label smoothing `0.05`, batch size 32, at most 300 epochs, and patience 40. It saves five checkpoints and fold-specific feature standardization under `models/`. Expected OOF macro-F1 is `0.5309`.

### 3. Export held-out embeddings and run the 25-member MLP ensemble

~~~bash
python scripts/evaluate_classifier_heldout.py \
  --trainer nnUNetTrainerSubtypeHeadAdamW \
  --folds 0 1 2 3 4 \
  --checkpoint checkpoint_best_cls.pth \
  --save-classification-embeddings \
  --output predictions/reset_head_5fold_val_case_embeddings_no_tta \
  --run-name reset-head-5fold-heldout-case-embedding-export-no-tta \
  --wandb-group uhn-pancreas-encoder-embedding-head-audit-20260803

python scripts/evaluate_frozen_embedding_mlp_heldout.py \
  --models predictions/frozen_embedding_mlp16_5fold_oof \
  --embeddings predictions/reset_head_5fold_val_case_embeddings_no_tta/classification_embeddings \
  --output predictions/frozen_embedding_mlp16_heldout \
  --run-name encoder-meanmax-mlp16-heldout-no-tta \
  --wandb-group uhn-pancreas-encoder-embedding-head-audit-20260803
~~~

Each of five MLPs is evaluated on each of five encoder embeddings and the 25 probability vectors are averaged. Expected held-out macro-F1 is `0.5839`. Flip TTA is disabled for this branch because it reduced macro-F1 to `0.5241`.

### 4. Produce the cross-attention branch and fuse probabilities

~~~bash
python scripts/evaluate_classifier_heldout.py \
  --trainer nnUNetTrainerSubtypeHeadAdamW \
  --folds 0 1 2 3 4 \
  --checkpoint checkpoint_best_cls.pth \
  --tta \
  --output predictions/reset_head_5fold_val_case_embeddings_tta \
  --run-name original-cross-attention-5fold-heldout-8x-tta \
  --wandb-group neural-encoder-head-product-ensemble-20260803

python scripts/evaluate_probability_ensemble.py \
  --probabilities \
    predictions/reset_head_5fold_val_case_embeddings_tta/subtype_probabilities.json \
    predictions/frozen_embedding_mlp16_heldout/probabilities.json \
  --probability-keys root mlp \
  --weights 1 1 \
  --aggregation geometric \
  --split validation \
  --output predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout \
  --run-name original-tta-frozen-encoder-mlp16-equal-geometric-heldout \
  --wandb-group neural-encoder-head-product-ensemble-20260803
~~~

Fusion computes `normalize(sqrt(p_attention * p_mlp))`; it does not fit a validation-set weight, bias, threshold, or meta-classifier. Expected held-out macro-F1 is `0.6296`.

For the unlabelled test set, export embeddings with `nnUNetv2_predict_quiz --disable_tta --save_classification_embeddings`, run `evaluate_frozen_embedding_mlp_heldout.py --split test`, produce cross-attention probabilities with ordinary 8-view TTA, and run `evaluate_probability_ensemble.py --split test` with the same equal geometric weights.

## Inference

Run five-fold inference. Mirroring TTA and foreground-evidence-weighted patch pooling are enabled by default.

~~~bash
MODEL="$nnUNet_results/Dataset001_PancreasQuiz/nnUNetTrainerMultiTaskSubtype__nnUNetResEncUNetMPlans__3d_fullres"

nnUNetv2_predict_quiz -m "$MODEL" -i "$nnUNet_raw/Dataset001_PancreasQuiz/imagesVal" -o predictions/val -f 0 1 2 3 4
nnUNetv2_predict_quiz -m "$MODEL" -i "$nnUNet_raw/Dataset001_PancreasQuiz/imagesTs" -o predictions/test -f 0 1 2 3 4
~~~

Useful options are `-chk checkpoint_best.pth`, `-step_size 0.5`, `--disable_tta`, `--uniform_pooling`, and `-device {cuda,cpu,mps}`. Outputs include NIfTI segmentations, `subtype_results.csv`, and subtype probabilities in CSV and JSON.

No morphology-based postprocessing is used. Fold/view segmentation probabilities are averaged and converted to labels by argmax; neural patch probabilities are aggregated at case level.

## Evaluation

Evaluate and save machine-readable metrics:

~~~bash
nnUNetv2_evaluate_quiz -i predictions/val -d "$nnUNet_raw/Dataset001_PancreasQuiz" -o predictions/val/summary.json --track undergraduate
~~~

Use `--track master` for the stricter 0.91 whole-pancreas Dice, 0.31 lesion Dice, and 0.70 macro-F1 targets. The evaluator reports segmentation mean/standard deviation and classification MCC, balanced accuracy, macro-F1, per-class sensitivity/specificity/F1, confusion matrix, one-vs-rest AUROC, and average precision.

Package test predictions:

~~~bash
python - <<'PY'
from nnunetv2.evaluation.evaluate_quiz import package_submission
package_submission("predictions/test", "submissions/my_results.zip")
PY
~~~

## Tests

~~~bash
python -m pytest nnunetv2/tests/test_quiz_metrics.py nnunetv2/tests/test_multitask_logger.py nnunetv2/tests/test_multitask_patch_weighting.py nnunetv2/tests/test_resenc_unet_with_cls.py nnunetv2/tests/test_multitask_trainer.py nnunetv2/tests/test_quiz_inference.py nnunetv2/tests/test_frozen_embedding_mlp.py nnunetv2/tests/test_subtype_head_adamw_trainer.py nnunetv2/tests/test_lesion_centered_attention_trainer.py nnunetv2/tests/test_probability_ensemble_script.py -m "not slow" -q
~~~

Remove `-m "not slow"` to include tests that instantiate the full network or execute training steps on CPU.

## Documentation

| Document | Contents |
| --- | --- |
| [Documentation index](documentation/README.md) | Retained project documentation |
| [End-to-end workflow](documentation/pancreas_quiz_workflow.md) | Operational runbook |
| [Winning solution](documentation/final_report/winning_solution.md) | Accepted system and metrics |
| [Experiment summary](documentation/training_experiment_summary.md) | Main comparisons |
| [CT experiment record](documentation/rsna_ct_classification_experiments.md) | Chronological trials |
| [Final report evidence](documentation/final_report/README.md) | Metrics and artifact manifest |
| [Paper source](report_submission/paper.tex) | MICCAI-style LaTeX report |

Formal runs are recorded in `charlesg-chen-university-of-toronto/uhn-pancreas-quiz`; immutable IDs are catalogued in [wandb_runs.csv](documentation/final_report/data/wandb_runs.csv).

## Contributing and license

The repository uses the [Apache License 2.0](LICENSE), matching vendored nnU-Net. Create focused branches, update task-specific tests, and describe the split and evaluation protocol in pull requests. Do not commit task data, identifiable material, checkpoints, W&B credentials, or large prediction trees.

## Citation and acknowledgement

The paper citation will be added after publication. Please also cite nnU-Net:

~~~bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal={Nature Methods},
  volume={18},
  pages={203--211},
  year={2021}
}
~~~

We thank the UHN task organizers for the de-identified dataset and evaluation brief, and the nnU-Net contributors at DKFZ/Helmholtz Imaging for the underlying framework.
