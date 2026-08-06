#!/usr/bin/env python3
"""Run genuine fold-held-out sliding-window classification and log one OOF result to W&B."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import wandb

from nnunetv2.evaluation.quiz_metrics import classification_metrics, confusion_matrix_counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--folds', nargs='+', type=int, required=True)
    parser.add_argument('--checkpoint', default='checkpoint_best_cls.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--wandb-group', required=True)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--step-size', type=float, default=0.5,
                        help='sliding-window step size passed to nnUNet inference')
    parser.add_argument('--tta-axes', nargs='+', type=int, choices=(0, 1, 2),
                        help='optional mirror-axis subset: one axis=2x, two=4x, three=8x TTA')
    parser.add_argument(
        '--classification-patch-pooling',
        choices=('weighted_mean', 'top1', 'top3'),
        default='weighted_mean',
        help='How to combine sliding-window subtype predictions within each case.',
    )
    parser.add_argument(
        '--classification-gamma-tta', nargs='+', type=float,
        help='additional unmirrored, classification-only statistics-retaining gamma views',
    )
    parser.add_argument(
        '--uniform-pooling', action='store_true',
        help='Give every patch equal evidence weight instead of using predicted pancreas foreground.',
    )
    parser.add_argument(
        '--max-parallel-folds', type=int, default=1,
        help='Run independent fold predictors concurrently on the same GPU (default: 1).',
    )
    parser.add_argument('--save-classification-embeddings', action='store_true')
    parser.add_argument('--save-classification-patch-embeddings', action='store_true')
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
    if args.max_parallel_folds < 1:
        raise ValueError('--max-parallel-folds must be at least 1')
    if not 0 < args.step_size <= 1:
        raise ValueError('--step-size must be in (0, 1]')
    if args.tta_axes is not None and not args.tta:
        raise ValueError('--tta-axes requires --tta')
    primary = args.primary_repo.resolve()
    raw_dataset = primary / 'nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz'
    preprocessed = primary / 'nnUNet_data/nnUNet_preprocessed/Dataset001_PancreasQuiz'
    model = (primary / 'nnUNet_data/nnUNet_results/Dataset001_PancreasQuiz'
             / f'{args.trainer}__nnUNetResEncUNetMPlans__3d_fullres')
    splits = json.loads((preprocessed / 'splits_final.json').read_text())
    label_splits = json.loads((raw_dataset / 'subtype_labels.json').read_text())
    labels = {}
    for split_name, mapping in label_splits.items():
        if split_name != 'validation':
            labels.update(mapping)
    expected = []
    for fold in args.folds:
        expected.extend(splits[fold]['val'])
    if len(expected) != len(set(expected)):
        raise RuntimeError('requested folds have overlapping validation cases')
    for fold in args.folds:
        checkpoint = model / f'fold_{fold}' / args.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.output.exists():
        raise FileExistsError(f'archive existing OOF output before rerunning: {args.output}')
    args.output.mkdir(parents=True)

    predictor = Path(sys.executable).parent / 'nnUNetv2_predict_quiz'
    def predict_fold(fold):
        fold_input = args.output / '_fold_inputs' / f'fold_{fold}'
        fold_output = args.output / f'fold_{fold}'
        fold_input.mkdir(parents=True)
        for case in splits[fold]['val']:
            source = raw_dataset / 'imagesTr' / f'{case}_0000.nii.gz'
            if not source.is_file():
                raise FileNotFoundError(source)
            (fold_input / source.name).symlink_to(source)
        command = [
            str(predictor), '-m', str(model), '-i', str(fold_input), '-o', str(fold_output),
            '-f', str(fold), '-chk', args.checkpoint,
            '-step_size', str(args.step_size),
            '--classification_patch_pooling', args.classification_patch_pooling,
        ]
        if not args.tta:
            command.append('--disable_tta')
        elif args.tta_axes is not None:
            command.extend(['--tta_axes', *[str(axis) for axis in args.tta_axes]])
        if args.uniform_pooling:
            command.append('--uniform_pooling')
        if args.classification_gamma_tta:
            command.extend([
                '--classification_gamma_tta',
                *[str(gamma) for gamma in args.classification_gamma_tta],
            ])
        if args.save_classification_embeddings:
            command.append('--save_classification_embeddings')
        if args.save_classification_patch_embeddings:
            command.append('--save_classification_patch_embeddings')
        subprocess.run(command, check=True, env=os.environ.copy())
        probabilities = json.loads((fold_output / 'subtype_probabilities.json').read_text())
        if set(probabilities) != set(splits[fold]['val']):
            raise RuntimeError(f'fold {fold} prediction cases do not match its validation split')
        return fold, probabilities

    merged = {}
    workers = min(args.max_parallel_folds, len(args.folds))
    if workers == 1:
        fold_results = [predict_fold(fold) for fold in args.folds]
    else:
        fold_results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(predict_fold, fold): fold for fold in args.folds}
            for future in as_completed(futures):
                fold_results.append(future.result())
    for fold, probabilities in sorted(fold_results):
        overlap = set(merged) & set(probabilities)
        if overlap:
            raise RuntimeError(
                f'fold {fold} produced cases already predicted by another fold: '
                f'{sorted(overlap)[:3]}'
            )
        merged.update(probabilities)

    if set(merged) != set(expected):
        raise RuntimeError('merged OOF predictions do not cover every expected case exactly once')
    ordered = sorted(expected)
    probabilities = np.asarray([merged[case] for case in ordered], dtype=np.float64)
    targets = np.asarray([labels[case] for case in ordered], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(targets, probabilities, num_classes=3, prefix='oof/')
    metrics = {key: scalarize(value) for key, value in metrics.items()}
    metrics['oof/confusion_matrix'] = confusion_matrix_counts(targets, predictions).tolist()
    metrics['oof/n_cases'] = len(ordered)
    metrics['oof/folds'] = list(args.folds)

    (args.output / 'subtype_probabilities.json').write_text(json.dumps(merged, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(metrics, indent=2))
    with (args.output / 'subtype_probabilities.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Names', 'p_subtype0', 'p_subtype1', 'p_subtype2'])
        for case in ordered:
            writer.writerow([f'{case}.nii.gz', *merged[case]])
    with (args.output / 'subtype_results.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Names', 'Subtype'])
        for case, prediction in zip(ordered, predictions):
            writer.writerow([f'{case}.nii.gz', int(prediction)])

    run = wandb.init(
        project=os.environ.get('nnUNet_wandb_project', 'uhn-pancreas-quiz'),
        entity=os.environ.get('WANDB_ENTITY') or None,
        group=args.wandb_group,
        name=args.run_name,
        job_type='internal-oof-evaluation',
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
            'foreground_weighted_pooling': not args.uniform_pooling,
            'max_parallel_folds': workers,
            'save_classification_embeddings': args.save_classification_embeddings,
            'save_classification_patch_embeddings': args.save_classification_patch_embeddings,
            'evaluation': 'fold-held-out full-volume sliding-window OOF',
            'backbone_training': False,
            'segmentation_head_training': False,
            'input': 'processed_3d_ct_only',
            'excluded_inputs': ['morphology', 'case_level_features'],
        },
    )
    scalar_metrics = {key: value for key, value in metrics.items()
                      if isinstance(value, (int, float))}
    run.log(scalar_metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f'{args.run_name}-predictions', type='oof-predictions')
    artifact.add_file(str(args.output / 'summary.json'))
    artifact.add_file(str(args.output / 'subtype_probabilities.csv'))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
