#!/usr/bin/env python3
"""Evaluate a prespecified arithmetic or geometric ensemble of CT-neural probabilities.

Weights must be supplied before evaluation; this script deliberately performs no optimization on
the evaluation labels. It is used for cheap OOF-selected checkpoint/head blends after the expensive
3D sliding-window predictions already exist.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from nnunetv2.evaluation.quiz_metrics import (
    classification_metrics,
    confusion_matrix_counts,
)
from nnunetv2.inference.predict_quiz import write_subtype_outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--probability-keys", nargs="+",
        help="one JSON key per file, or 'root' when the file itself is the case mapping",
    )
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument(
        "--aggregation", choices=("arithmetic", "geometric"), default="arithmetic",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument(
        "--primary-repo", type=Path, default=Path.cwd()
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.weights is None:
        weights = np.ones(len(args.probabilities), dtype=np.float64)
    else:
        if len(args.weights) != len(args.probabilities):
            raise ValueError("--weights must match --probabilities")
        weights = np.asarray(args.weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive total")
    weights /= weights.sum()
    if args.probability_keys is None:
        probability_keys = ["root"] * len(args.probabilities)
    else:
        if len(args.probability_keys) != len(args.probabilities):
            raise ValueError("--probability-keys must match --probabilities")
        probability_keys = list(args.probability_keys)

    labels_path = (
        args.primary_repo
        / "nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz/subtype_labels.json"
    )
    label_splits = json.loads(labels_path.read_text())
    if args.split == "test":
        labels = None
        expected_cases = {
            path.name[:-len("_0000.nii.gz")]
            for path in (
                args.primary_repo
                / "nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz/imagesTs"
            ).glob("*_0000.nii.gz")
        }
    else:
        labels = label_splits[args.split]
        expected_cases = set(labels)
    components = []
    for path, key in zip(args.probabilities, probability_keys):
        payload = json.loads(path.read_text())
        if key != "root":
            if key not in payload or not isinstance(payload[key], dict):
                raise RuntimeError(f"{path} does not contain probability mapping key {key!r}")
            payload = payload[key]
        components.append(payload)
    for path, component in zip(args.probabilities, components):
        if set(component) != expected_cases:
            missing = sorted(expected_cases - set(component))[:5]
            extra = sorted(set(component) - expected_cases)[:5]
            raise RuntimeError(f"case mismatch for {path}: missing={missing}, extra={extra}")

    cases = sorted(expected_cases)
    component_arrays = np.stack([
        np.asarray([component[case] for case in cases], dtype=np.float64)
        for component in components
    ])
    if component_arrays.shape != (len(components), len(cases), 3):
        raise RuntimeError(f"invalid probability shape {component_arrays.shape}")
    if not np.isfinite(component_arrays).all() or (component_arrays < 0).any():
        raise RuntimeError("probabilities must be finite and non-negative")
    component_arrays /= component_arrays.sum(axis=2, keepdims=True).clip(1e-12)
    if args.aggregation == "arithmetic":
        probabilities = np.tensordot(weights, component_arrays, axes=(0, 0))
    else:
        # A weighted product of experts is the weighted mean in log-probability space. Equal
        # weights introduce no fitted calibration parameter and reward agreement between heads.
        log_probabilities = np.log(component_arrays.clip(1e-12))
        fused_logits = np.tensordot(weights, log_probabilities, axes=(0, 0))
        fused_logits -= fused_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(fused_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(1e-12)
    if labels is None:
        metrics = {"test/n_cases": len(cases)}
    else:
        targets = np.asarray([labels[case] for case in cases], dtype=np.int64)
        prefix = "oof/" if args.split == "train" else "cls/"
        metrics = classification_metrics(targets, probabilities, 3, prefix=prefix)
        metrics[f"{prefix}confusion_matrix"] = confusion_matrix_counts(
            targets, probabilities.argmax(1)
        ).tolist()
        metrics[f"{prefix}n_cases"] = len(cases)
        metrics = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in metrics.items()
        }

    args.output.mkdir(parents=True)
    probability_mapping = {
        case: probability for case, probability in zip(cases, probabilities)
    }
    write_subtype_outputs(probability_mapping, str(args.output))
    (args.output / "summary.json").write_text(json.dumps(metrics, indent=2))

    run = wandb.init(
        project=os.environ.get("nnUNet_wandb_project", "uhn-pancreas-quiz"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        group=args.wandb_group,
        name=args.run_name,
        job_type="prespecified-probability-ensemble-evaluation",
        config={
            "probability_files": [str(path.resolve()) for path in args.probabilities],
            "probability_keys": probability_keys,
            "weights": weights.tolist(),
            "aggregation": args.aggregation,
            "weight_selection": "prespecified_without_evaluation_label_optimization",
            "evaluation_split": args.split,
            "input": "processed_3d_ct_neural_predictions_only",
            "excluded_inputs": [
                "morphology", "case_level_engineered_features", "tabular_features"
            ],
            "external_pretrained_weights": False,
        },
    )
    run.log({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f"{args.run_name}-predictions", type="ensemble-predictions")
    for filename in (
        "summary.json", "subtype_probabilities.json", "subtype_probabilities.csv",
        "subtype_results.csv",
    ):
        artifact.add_file(str(args.output / filename))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
