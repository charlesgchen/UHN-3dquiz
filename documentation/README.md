# Project documentation

This directory contains only the pancreas quiz workflow, experiment record, and report evidence. Generic nnU-Net manuals were removed because this repository vendors nnU-Net as an implementation dependency rather than serving as a documentation mirror. For framework guidance, use the [official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation).

## Start here

| Document | Purpose |
| --- | --- |
| [../readme.md](../readme.md) | MICCAI-CODE-oriented installation and reproduction guide |
| [pancreas_quiz_workflow.md](pancreas_quiz_workflow.md) | Conversion, training, inference, and evaluation runbook |
| [final_report/winning_solution.md](final_report/winning_solution.md) | Accepted architecture, protocol, and results |
| [training_experiment_summary.md](training_experiment_summary.md) | Main experiment comparison |
| [rsna_ct_classification_experiments.md](rsna_ct_classification_experiments.md) | CT-only experiment history and decisions |
| [final_report/augmentation_ablation.md](final_report/augmentation_ablation.md) | Training augmentation and TTA ablations |
| [final_report/README.md](final_report/README.md) | Metrics, W&B registry, and artifact manifest |

The LaTeX manuscript and architecture reference are in `../report_submission/`.
