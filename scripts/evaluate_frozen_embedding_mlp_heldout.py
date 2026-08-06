#!/usr/bin/env python3
"""Run frozen nnU-Net embedding MLPs on held-out or unlabeled test CT scans.

The classifier models are fitted only to complete OOF embeddings from the 252 training cases. Each
held-out scan has one embedding from each of the five frozen nnU-Net folds. We average predictions
over both sources of model uncertainty: five MLP CV members and five nnU-Net embedding members.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb

from nnunetv2.evaluation.quiz_metrics import classification_metrics, confusion_matrix_counts
if __package__:
    from .train_frozen_embedding_mlp import EmbeddingMLP
else:
    from train_frozen_embedding_mlp import EmbeddingMLP


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models', type=Path, nargs='+', required=True,
        help='one or more directories containing models/fold_0.pth through fold_4.pth',
    )
    parser.add_argument('--embeddings', type=Path, required=True,
                        help='held-out classification_embeddings directory')
    parser.add_argument('--baseline-probabilities', type=Path,
                        help='optional frozen-head probability JSON for a prespecified equal blend')
    parser.add_argument(
        '--aux-probabilities', type=Path,
        help='optional neural probability JSON whose log probabilities were appended during MLP training',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--wandb-group', required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--feature-slice', choices=('both', 'mean', 'max'), default='both')
    parser.add_argument(
        '--member-pairing', choices=('cross-product', 'aligned'), default='cross-product',
        help='aligned pairs each MLP fold with the matching nnU-Net embedding member',
    )
    parser.add_argument('--primary-repo', type=Path, default=Path.cwd())
    return parser.parse_args()


def scalarize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def metric_dict(targets, probabilities, prefix):
    metrics = classification_metrics(targets, probabilities, num_classes=3, prefix=prefix)
    metrics[f'{prefix}confusion_matrix'] = confusion_matrix_counts(
        targets, probabilities.argmax(axis=1)).tolist()
    return {key: scalarize(value) for key, value in metrics.items()}


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    raw = args.primary_repo / 'nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz'
    label_splits = json.loads((raw / 'subtype_labels.json').read_text())
    if args.split == 'validation':
        labels = label_splits['validation']
        expected_cases = set(labels)
    else:
        labels = None
        test_folder = raw / 'imagesTs'
        expected_cases = {
            path.name[:-len('_0000.nii.gz')]
            for path in test_folder.glob('*_0000.nii.gz')
        }
    embedding_paths = sorted(args.embeddings.glob('*.npy'))
    cases = [path.stem for path in embedding_paths]
    if len(cases) != len(expected_cases) or len(set(cases)) != len(cases) or set(cases) != expected_cases:
        raise RuntimeError(
            f'expected exactly {len(expected_cases)} {args.split} embeddings, found {len(cases)}'
        )

    embeddings = np.stack([np.load(path) for path in embedding_paths]).astype(np.float32)
    if embeddings.shape[:2] != (len(cases), 5) or embeddings.ndim != 3:
        raise RuntimeError(
            f'expected [{len(cases)}, 5, D] embeddings, got {embeddings.shape}'
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError('embeddings contain non-finite values')
    if args.feature_slice == 'mean':
        embeddings = embeddings[:, :, :embeddings.shape[2] // 2]
    elif args.feature_slice == 'max':
        embeddings = embeddings[:, :, embeddings.shape[2] // 2:]
    auxiliary_dim = 0
    if args.aux_probabilities is not None:
        auxiliary_mapping = json.loads(args.aux_probabilities.read_text())
        if set(auxiliary_mapping) != set(cases):
            raise RuntimeError('--aux-probabilities do not cover the evaluation cases exactly')
        auxiliary = np.asarray(
            [auxiliary_mapping[case] for case in cases], dtype=np.float32
        )
        if (auxiliary.shape != (len(cases), 3) or not np.isfinite(auxiliary).all()
                or (auxiliary < 0).any()):
            raise RuntimeError(f'invalid auxiliary neural probabilities {auxiliary.shape}')
        auxiliary /= auxiliary.sum(axis=1, keepdims=True).clip(1e-8)
        auxiliary = np.log(auxiliary.clip(1e-6))
        auxiliary = np.repeat(auxiliary[:, None, :], embeddings.shape[1], axis=1)
        embeddings = np.concatenate([embeddings, auxiliary], axis=2)
        auxiliary_dim = 3

    member_probabilities = []
    for model_set in args.models:
        model_paths = sorted((model_set / 'models').glob('fold_*.pth'))
        if len(model_paths) != 5:
            raise RuntimeError(f'expected five MLP checkpoints in {model_set}, found {len(model_paths)}')
        for model_index, path in enumerate(model_paths):
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            mean = np.asarray(checkpoint['mean'], dtype=np.float32)
            std = np.asarray(checkpoint['std'], dtype=np.float32)
            if mean.shape != (embeddings.shape[2],) or std.shape != mean.shape:
                raise RuntimeError(
                    f'{path} normalization shape {mean.shape} != {(embeddings.shape[2],)}'
                )
            model = EmbeddingMLP(
                embeddings.shape[2], int(checkpoint['hidden_dim']),
                float(checkpoint['dropout']))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            if args.member_pairing == 'aligned':
                member = checkpoint.get('embedding_member')
                if member is None or int(member) != model_index:
                    raise RuntimeError(
                        f'{path} is not an aligned member-{model_index} MLP checkpoint'
                    )
                normalized = (
                    embeddings[:, model_index, :] - mean[None, :]
                ) / std[None, :]
                with torch.inference_mode():
                    probabilities = model(torch.from_numpy(normalized)).softmax(1).numpy()
            else:
                normalized = (embeddings - mean[None, None, :]) / std[None, None, :]
                with torch.inference_mode():
                    logits = model(
                        torch.from_numpy(normalized.reshape(-1, embeddings.shape[2]))
                    )
                    probabilities = logits.softmax(1).numpy().reshape(
                        len(cases), 5, 3
                    ).mean(axis=1)
            member_probabilities.append(probabilities)
    probabilities = np.mean(member_probabilities, axis=0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = None if labels is None else np.asarray(
        [labels[case] for case in cases], dtype=np.int64
    )

    metrics = {} if targets is None else metric_dict(targets, probabilities, 'mlp/')
    outputs = {'mlp': {case: [float(value) for value in probability]
                       for case, probability in zip(cases, probabilities)}}
    if args.baseline_probabilities:
        baseline_map = json.loads(args.baseline_probabilities.read_text())
        if set(baseline_map) != set(cases):
            raise RuntimeError('baseline probabilities do not cover the held-out cases exactly')
        baseline = np.asarray([baseline_map[case] for case in cases], dtype=np.float32)
        blended = 0.5 * probabilities + 0.5 * baseline
        blended /= blended.sum(axis=1, keepdims=True)
        if targets is not None:
            metrics.update(metric_dict(targets, blended, 'equal_blend/'))
        outputs['equal_blend'] = {
            case: [float(value) for value in probability]
            for case, probability in zip(cases, blended)
        }

    metrics['n_cases'] = len(cases)
    (args.output / 'summary.json').write_text(json.dumps(metrics, indent=2))
    (args.output / 'probabilities.json').write_text(json.dumps(outputs, indent=2))
    with (args.output / 'mlp_probabilities.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Names', 'p_subtype0', 'p_subtype1', 'p_subtype2'])
        for case, probability in zip(cases, probabilities):
            writer.writerow([f'{case}.nii.gz', *probability.tolist()])

    run = wandb.init(
        project=os.environ.get('nnUNet_wandb_project', 'uhn-pancreas-quiz'),
        entity=os.environ.get('WANDB_ENTITY') or None,
        group=args.wandb_group,
        name=args.run_name,
        job_type=(
            'frozen-embedding-mlp-heldout-evaluation'
            if args.split == 'validation' else 'frozen-embedding-mlp-test-inference'
        ),
        config={
            'input': 'five_member_frozen_nnunet_bottleneck_embeddings_from_processed_3d_ct',
            'auxiliary_neural_probability_dim': auxiliary_dim,
            'auxiliary_probability_file': (
                str(args.aux_probabilities.resolve()) if args.aux_probabilities else None
            ),
            'auxiliary_probability_transform': 'log' if auxiliary_dim else None,
            'embedding_members': 5,
            'mlp_model_sets': len(args.models),
            'mlp_members': 5 * len(args.models),
            'member_pairing': args.member_pairing,
            'aggregation': (
                f'mean_probabilities_over_{5 * len(args.models)}_aligned_encoder_mlp_pairs'
                if args.member_pairing == 'aligned'
                else f'mean_probabilities_over_{25 * len(args.models)}_member_cross_product'
            ),
            'feature_slice': args.feature_slice,
            'evaluation_split': args.split,
            'baseline_equal_blend': bool(args.baseline_probabilities),
            'backbone_training': False,
            'segmentation_head_training': False,
            'external_pretrained_weights': False,
            'excluded_inputs': [
                'morphology', 'shape', 'volume', 'coordinates', 'case_level_tabular_features'],
        },
    )
    run.log({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f'{args.run_name}-predictions', type='heldout-predictions')
    artifact.add_dir(str(args.output))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
