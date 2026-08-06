#!/usr/bin/env python3
"""Apply stable, human-readable labels to the final-report W&B runs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import wandb


ENTITY = "charlesg-chen-university-of-toronto"
PROJECT = "uhn-pancreas-quiz"


@dataclass(frozen=True)
class RunLabel:
    display_name: str
    group: str


LABELS = {
    "x0pcluox": RunLabel("Original joint multitask | Fold 0", "Original joint model - 5 folds"),
    "fffvqm32": RunLabel("Original joint multitask | Fold 1", "Original joint model - 5 folds"),
    "22ic3abr": RunLabel("Original joint multitask | Fold 2", "Original joint model - 5 folds"),
    "jqylrdqi": RunLabel("Original joint multitask | Fold 3", "Original joint model - 5 folds"),
    "fm4cg0i0": RunLabel("Original joint multitask | Fold 4", "Original joint model - 5 folds"),
    "hyt8k0qs": RunLabel(
        "Retrained cross-attention head | Fold 0", "Retrained cross-attention head - 5 folds"
    ),
    "sjzld1g6": RunLabel(
        "Retrained cross-attention head | Fold 1", "Retrained cross-attention head - 5 folds"
    ),
    "6hxhbmyl": RunLabel(
        "Retrained cross-attention head | Fold 2", "Retrained cross-attention head - 5 folds"
    ),
    "7a8g6y3y": RunLabel(
        "Retrained cross-attention head | Fold 3", "Retrained cross-attention head - 5 folds"
    ),
    "6r7k05q7": RunLabel(
        "Retrained cross-attention head | Fold 4", "Retrained cross-attention head - 5 folds"
    ),
    "i1jb6pvz": RunLabel(
        "Encoder mean-max MLP | 5-fold OOF", "Encoder mean-max MLP - OOF"
    ),
    "jl9wh4bc": RunLabel(
        "Retrained cross-attention | Held-out | 8x TTA",
        "Retrained cross-attention - held-out 8x TTA",
    ),
    "mwfad9f2": RunLabel(
        "Final CT fusion | Held-out", "Final fusion - held-out"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist labels. Without this flag, only print the proposed changes.",
    )
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ and "WAND_API_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WAND_API_KEY"]

    api = wandb.Api()
    for run_id, target in LABELS.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        print(
            f"{run_id}: {run.name!r} / {run.group!r} -> "
            f"{target.display_name!r} / {target.group!r}"
        )
        if not args.apply:
            continue
        run.name = target.display_name
        # Public Run.update persists `groupName`, but the current SDK does not
        # expose a group setter. This is the backing field consumed by update().
        run._attrs["group"] = target.group  # noqa: SLF001
        run.update()


if __name__ == "__main__":
    main()
