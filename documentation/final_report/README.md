# Final report workspace

Last consolidated: 2026-08-06 UTC

This directory is the report-ready source of truth for the pancreas quiz experiments. It keeps
the concise narrative and lightweight, machine-readable evidence together in the `master`
worktree. Large checkpoints, NIfTI predictions, and raw W&B files remain in their existing
locations and are referenced rather than duplicated.

## Final result

| Metric | Held-out result | Undergraduate requirement | Status |
|---|---:|---:|---|
| Whole-pancreas DSC | **0.9325** | 0.90 | Pass |
| Pancreas-lesion DSC | **0.6308** | 0.27 | Pass |
| Classification macro-F1 | **0.6296** | 0.60 | Pass |
| Classification MCC | **0.4905** | Not thresholded | Reported |

The accepted classifier is an equal geometric fusion of the original five-fold cross-attention
head with 8-view flip TTA and a five-fold, 16-unit encoder mean/max MLP without TTA. It consumes
only neural features from the supplied processed 3D CT scans. It does not use morphology,
engineered case features, tabular classifiers, external datasets, or external pretrained weights.

## Where to start

| File | Purpose |
|---|---|
| [winning_solution.md](winning_solution.md) | Final architecture, inference, metrics, and recommended report wording |
| [augmentation_ablation.md](augmentation_ablation.md) | What was tried for training augmentation and TTA, with exact outcomes |
| [experiment_metrics.csv](data/experiment_metrics.csv) | Normalized metrics for the main result and report-critical ablations |
| [metrics.json](data/metrics.json) | Full machine-readable accepted Metrics Reloaded result plus matched pooling ablations |
| [wandb_runs.csv](data/wandb_runs.csv) | W&B provenance for the winning system and major ablations |
| [artifact_manifest.csv](data/artifact_manifest.csv) | Repository paths, roles, hashes, and retention policy |
| [source_summaries/README.md](data/source_summaries/README.md) | Content-equivalent snapshots of report-critical source metric JSON files |
| [create_wandb_view.py](create_wandb_view.py) | Reproducible definition of the compact final-report W&B saved view |
| [rename_wandb_runs.py](rename_wandb_runs.py) | Idempotent mapping from immutable W&B run IDs to readable display and group names |
| [../training_experiment_summary.md](../training_experiment_summary.md) | Short overall experiment summary |
| [../rsna_ct_classification_experiments.md](../rsna_ct_classification_experiments.md) | Detailed chronological experiment record, now copied into `master` |
| [../pancreas_quiz_workflow.md](../pancreas_quiz_workflow.md) | Original end-to-end nnU-Net workflow |

## W&B presentation

Use the manual saved view [Final Report - Core Metrics](https://wandb.ai/charlesg-chen-university-of-toronto/uhn-pancreas-quiz?nw=0imnn6sk47d),
not the project's automatic workspace. The automatic workspace creates a panel for every logged key and
is intentionally left unchanged so no raw history is lost.

The saved view exposes only the evidence requested by `ReadMe.pdf`:

- classification and segmentation training/validation loss curves;
- a clearly labeled sampled-case classification diagnostic and per-label segmentation validation
  Dice during training;
- an authoritative complete-volume classification table with macro-F1, MCC, and balanced accuracy;
  and
- a final segmentation scorecard with whole-pancreas Dice, lesion Dice, and lesion F2.

The per-epoch `val_case_cls/macro_f1` curve is retained only as a learning-dynamics diagnostic. It
averages sampled validation patches over a varying subset of internal-fold cases, so it can reveal
convergence or overfitting but cannot rank final models. The comparison table uses exact inference
on the same 36 complete held-out volumes and supplies the reportable classification scores.

The encoder-MLP fold curves are retained in a collapsed supporting section. The complete Metrics
Reloaded table, per-class results, confusion matrix, AUROC, average precision, and mean/SD details
belong in the written report rather than separate top-level W&B panels.

## Evaluation convention

OOF and held-out metrics answer different questions and must remain separate in the report:

- **Five-fold OOF (252 cases):** each case is predicted only by the model whose training fold
  excluded it. Use this for model-development comparisons and ablations.
- **Provided held-out validation (36 cases):** predictions use the complete five-fold ensemble.
  Use this for the final system result required by the brief.
- **Test (72 cases):** labels are unavailable. Report artifact validation and provenance, not a
  test macro-F1.

Because many decisions were explored, the held-out score has some model-selection optimism. The
report should present both OOF and held-out values, clearly label the split for every number, and
avoid describing the 36-case result as an unbiased estimate of deployment performance.

## Large artifact locations

- Accepted held-out predictions:
  `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout/`
- Locked test predictions:
  `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_test/`
- Final archive: `submissions/charles_chen_results.zip`
- Final archive SHA-256:
  `5403b303206f30898b93ba346d5265d85046a64f69cc04875a64ceed2ec0c78f`
- Final matched GAP logs: `run_logs/matched-gap-head-ablation-20260804/`
- nnU-Net checkpoints: `nnUNet_data/nnUNet_results/`

The `predictions/`, `run_logs/`, and checkpoint trees are intentionally not copied into this
directory. They are hundreds of megabytes, include intermediates, and would make the report data
harder to audit. The CSV manifest identifies the small subset that supports the final claims.
