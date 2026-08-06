#!/usr/bin/env python3
"""Create the compact W&B saved view used for the final report.

This intentionally creates a manual saved view instead of changing the project's
automatic workspace. All logged metrics remain available, while the shared view
contains only the evidence required by ReadMe.pdf plus a small final scorecard.

Requirements:
    pip install wandb wandb-workspaces

Authentication is read from ``WANDB_API_KEY``. For compatibility with this
project's existing ``.env``, ``WAND_API_KEY`` is also accepted.
"""

from __future__ import annotations

import argparse
import os

import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws


ENTITY = "charlesg-chen-university-of-toronto"
PROJECT = "uhn-pancreas-quiz"

SOURCE_NNUNET_RUNS = [
    "x0pcluox",
    "fffvqm32",
    "22ic3abr",
    "jqylrdqi",
    "fm4cg0i0",
]
CROSS_ATTENTION_RUNS = [
    "hyt8k0qs",
    "sjzld1g6",
    "6hxhbmyl",
    "7a8g6y3y",
    "6r7k05q7",
]
MLP_RUN = "i1jb6pvz"
SEGMENTATION_VALIDATION_RUN = "jl9wh4bc"
FINAL_CLASSIFICATION_RUN = "mwfad9f2"

VISIBLE_RUNS = [
    *SOURCE_NNUNET_RUNS,
    *CROSS_ATTENTION_RUNS,
    MLP_RUN,
    SEGMENTATION_VALIDATION_RUN,
    FINAL_CLASSIFICATION_RUN,
]


def build_workspace() -> ws.Workspace:
    fold_line_options = {
        "groupby_aggfunc": "mean",
        "groupby_rangefunc": "stddev",
        "smoothing_type": "none",
        "max_runs_to_show": 20,
    }

    required_training = ws.Section(
        name="Required training and validation evidence",
        is_open=True,
        pinned=True,
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=2),
        panels=[
            wr.LinePlot(
                title="Classification loss (train and validation)",
                y=["train_losses_cls", "val_losses_cls"],
                title_x="Epoch",
                title_y="Loss",
                **fold_line_options,
            ),
            wr.LinePlot(
                title="Segmentation loss (train and validation)",
                y=["train_losses_seg", "val_losses_seg"],
                title_x="Epoch",
                title_y="Loss",
                **fold_line_options,
            ),
            wr.LinePlot(
                title="Classification validation: macro-average F1",
                y=["val_case_cls/macro_f1"],
                range_y=(0, 1),
                title_x="Epoch",
                title_y="Macro-average F1",
                **fold_line_options,
            ),
            wr.LinePlot(
                title="Segmentation validation: per-label Dice",
                y=[
                    "dice_per_class_or_region/class_1",
                    "dice_per_class_or_region/class_2",
                ],
                range_y=(0, 1),
                title_x="Epoch",
                title_y="Dice score",
                **fold_line_options,
            ),
        ],
    )

    final_validation = ws.Section(
        name="Final held-out validation scorecard",
        is_open=True,
        pinned=True,
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
        panels=[
            wr.BarPlot(
                title="Classification: macro-F1, MCC, and balanced accuracy",
                metrics=[
                    "cls/macro_f1",
                    "cls/mcc",
                    "cls/balanced_accuracy",
                ],
                orientation="h",
                range_x=(0, 1),
                max_runs_to_show=10,
            ),
            wr.BarPlot(
                title="Segmentation: whole/lesion Dice and lesion F2",
                metrics=[
                    "seg/dice_whole_pancreas_mean",
                    "seg/dice_lesion_mean",
                    "seg/f2_lesion_mean",
                ],
                orientation="h",
                range_x=(0, 1),
                max_runs_to_show=10,
            ),
        ],
    )

    supporting_mlp = ws.Section(
        name="Supporting encoder-MLP branch (collapsed)",
        is_open=False,
        pinned=False,
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
        panels=[
            wr.LinePlot(
                title="MLP five-fold training loss",
                y=[f"cv_fold_{fold}/train_loss" for fold in range(5)],
                title_x="Epoch",
                title_y="Cross-entropy loss",
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="MLP five-fold validation macro-average F1",
                y=[f"cv_fold_{fold}/macro_f1" for fold in range(5)],
                range_y=(0, 1),
                title_x="Epoch",
                title_y="Macro-average F1",
                smoothing_type="none",
            ),
        ],
    )

    return ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name="Final Report - Core Metrics",
        auto_generate_panels=False,
        sections=[required_training, final_validation, supporting_mlp],
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            smoothing_type="none",
            ignore_outliers=False,
            sort_panels_alphabetically=False,
            group_by_prefix="last",
            max_runs=20,
        ),
        runset_settings=ws.RunsetSettings(
            filters=expr.And(
                expr.Metric("ID").isin(VISIBLE_RUNS),
                expr.Metric("State") == "finished",
            ),
            groupby=[expr.Metric("Group")],
            pinned_columns=[
                "summary:cls/macro_f1",
                "summary:cls/mcc",
                "summary:seg/dice_whole_pancreas_mean",
                "summary:seg/dice_lesion_mean",
            ],
            baseline_run=SEGMENTATION_VALIDATION_RUN,
            pinned_runs=[FINAL_CLASSIFICATION_RUN, SEGMENTATION_VALIDATION_RUN],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and serialize the view without saving it to W&B.",
    )
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ and "WAND_API_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WAND_API_KEY"]

    workspace = build_workspace()
    # Serialization is the strongest local validation available before the API write.
    workspace._to_model()  # noqa: SLF001
    if args.dry_run:
        print(
            f"Validated '{workspace.name}': "
            f"{len(workspace.sections)} sections, "
            f"{sum(len(section.panels) for section in workspace.sections)} panels, "
            f"{len(VISIBLE_RUNS)} runs."
        )
        return

    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
