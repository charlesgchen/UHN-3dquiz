#!/usr/bin/env python3
"""Run a complete held-out fold ensemble and log metrics plus predictions to W&B."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import wandb

from nnunetv2.evaluation.evaluate_quiz import check_quiz_targets, evaluate_quiz_validation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--checkpoint', default='checkpoint_best_cls.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--wandb-group', required=True)
    parser.add_argument('--track', choices=('undergraduate', 'master'), default='master',
                        help='README threshold track used only for pass/fail summary fields')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--step-size', type=float, default=0.5,
                        help='sliding-window step size passed to nnUNet inference')
    parser.add_argument('--tta-axes', nargs='+', type=int, choices=(0, 1, 2),
                        help='optional mirror-axis subset: one axis=2x, two=4x, three=8x TTA')
    parser.add_argument('--classification-patch-pooling', default='weighted_mean',
                        choices=('weighted_mean', 'top1', 'top3'))
    parser.add_argument(
        '--classification-gamma-tta', nargs='+', type=float,
        help='additional unmirrored, classification-only statistics-retaining gamma views',
    )
    parser.add_argument('--save-classification-embeddings', action='store_true',
                        help='save one frozen bottleneck mean/max embedding per nnU-Net fold')
    parser.add_argument('--save-classification-patch-embeddings', action='store_true',
                        help='save every frozen patch embedding and its predicted-lesion evidence')
    parser.add_argument('--primary-repo', type=Path, default=Path.cwd())
    return parser.parse_args()


def scalarize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    args = parse_args()
    if args.tta_axes is not None and not args.tta:
        raise ValueError('--tta-axes requires --tta')
    if not 0 < args.step_size <= 1:
        raise ValueError('--step-size must be in (0, 1]')
    primary = args.primary_repo.resolve()
    dataset = primary / 'nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz'
    model = (primary / 'nnUNet_data/nnUNet_results/Dataset001_PancreasQuiz'
             / f'{args.trainer}__nnUNetResEncUNetMPlans__3d_fullres')
    for fold in args.folds:
        checkpoint = model / f'fold_{fold}' / args.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.output.exists():
        raise FileExistsError(f'archive existing evaluation output before rerunning: {args.output}')

    predictor = Path(sys.executable).parent / 'nnUNetv2_predict_quiz'
    command = [
        str(predictor), '-m', str(model), '-i', str(dataset / 'imagesVal'),
        '-o', str(args.output), '-f', *[str(fold) for fold in args.folds],
        '-chk', args.checkpoint,
        '-step_size', str(args.step_size),
        '--classification_patch_pooling', args.classification_patch_pooling,
    ]
    if not args.tta:
        command.append('--disable_tta')
    elif args.tta_axes is not None:
        command.extend(['--tta_axes', *[str(axis) for axis in args.tta_axes]])
    if args.save_classification_embeddings:
        command.append('--save_classification_embeddings')
    if args.save_classification_patch_embeddings:
        command.append('--save_classification_patch_embeddings')
    if args.classification_gamma_tta:
        command.extend([
            '--classification_gamma_tta',
            *[str(gamma) for gamma in args.classification_gamma_tta],
        ])
    subprocess.run(command, check=True, env=os.environ.copy())

    summary_path = args.output / 'summary.json'
    metrics = evaluate_quiz_validation(str(args.output), str(dataset), str(summary_path))
    metrics = {key: scalarize(value) for key, value in metrics.items()}
    passes = check_quiz_targets(metrics, track=args.track)
    metrics.update({f'{args.track}_target/{key}': passed for key, passed in passes.items()})
    summary_path.write_text(json.dumps(metrics, indent=2))

    run = wandb.init(
        project=os.environ.get('nnUNet_wandb_project', 'uhn-pancreas-quiz'),
        entity=os.environ.get('WANDB_ENTITY') or None,
        group=args.wandb_group,
        name=args.run_name,
        job_type='heldout-evaluation',
        config={
            'trainer': args.trainer,
            'folds': list(args.folds),
            'checkpoint': args.checkpoint,
            'tta': args.tta,
            'sliding_window_step_size': args.step_size,
            'tta_axes': args.tta_axes,
            'tta_views': 1 if not args.tta else 2 ** len(args.tta_axes or (0, 1, 2)),
            'classification_patch_pooling': args.classification_patch_pooling,
            'classification_gamma_tta': args.classification_gamma_tta or [],
            'classification_total_tta_views': (
                (1 if not args.tta else 2 ** len(args.tta_axes or (0, 1, 2)))
                + len(args.classification_gamma_tta or [])
            ),
            'save_classification_embeddings': args.save_classification_embeddings,
            'save_classification_patch_embeddings': args.save_classification_patch_embeddings,
            'evaluation': 'complete_heldout_full_volume_sliding_window_ensemble',
            'threshold_track': args.track,
            'backbone_training': False,
            'segmentation_head_training': False,
            'input': 'processed_3d_ct_only',
            'excluded_inputs': ['morphology', 'case_level_features'],
        },
    )
    run.log({key: value for key, value in metrics.items()
             if isinstance(value, (int, float, bool))})
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f'{args.run_name}-predictions', type='heldout-predictions')
    for filename in ('summary.json', 'subtype_probabilities.csv', 'subtype_results.csv'):
        artifact.add_file(str(args.output / filename))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
