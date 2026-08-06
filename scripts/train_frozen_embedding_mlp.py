#!/usr/bin/env python3
"""Cross-validate a small MLP on CT-only frozen nnU-Net OOF embeddings.

The default stratified split is retained for reproducibility of the first screen. ``--split-mode
nnunet`` instead uses the exact five nnU-Net train/validation partitions. That mode tests whether
aligning classification-head validation with the backbone's OOF partition reduces the feature-
distribution mismatch without ever exposing a case to a classifier that trained on its label.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.model_selection import StratifiedKFold

from nnunetv2.evaluation.quiz_metrics import classification_metrics, confusion_matrix_counts


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.network = (nn.Linear(input_dim, 3) if hidden_dim == 0 else nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3)))

    def forward(self, x):
        return self.network(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--wandb-group', required=True)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--feature-noise-std', type=float, default=0.0)
    parser.add_argument('--feature-dropout', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument(
        '--class-weight-power', type=float, default=1.0,
        help='Exponent applied to inverse-frequency CE weights; 0 disables class weighting.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--min-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20260802)
    parser.add_argument('--feature-slice', choices=('both', 'mean', 'max'), default='both')
    parser.add_argument(
        '--aux-probabilities', type=Path,
        help='optional OOF neural probability JSON; log probabilities are appended to encoder features',
    )
    parser.add_argument('--split-mode', choices=('stratified', 'nnunet'), default='stratified')
    parser.add_argument(
        '--embedding-layout',
        choices=('oof', 'all-folds', 'all-folds-augmented'),
        default='oof',
        help=(
            'oof expects one fold-held-out embedding per case; all-folds expects five '
            'fold-member embeddings per case and trains each MLP on its matching member; '
            'all-folds-augmented uses every member as training-time feature augmentation '
            'but validates only on the matching held-out member'
        ),
    )
    parser.add_argument('--primary-repo', type=Path, default=Path('/workspace/app/UHN-3dquiz'))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    if args.feature_noise_std < 0:
        raise ValueError('--feature-noise-std must be non-negative')
    if not 0 <= args.feature_dropout < 1:
        raise ValueError('--feature-dropout must be in [0, 1)')
    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be positive')
    if args.weight_decay < 0:
        raise ValueError('--weight-decay must be non-negative')
    if not 0 <= args.label_smoothing < 1:
        raise ValueError('--label-smoothing must be in [0, 1)')
    if args.class_weight_power < 0:
        raise ValueError('--class-weight-power must be non-negative')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    if args.max_epochs < 1 or args.min_epochs < 0 or args.patience < 1:
        raise ValueError('epoch and patience settings are invalid')
    if args.min_epochs >= args.max_epochs:
        raise ValueError('--min-epochs must be smaller than --max-epochs')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / 'models').mkdir()

    raw = args.primary_repo / 'nnUNet_data/nnUNet_raw/Dataset001_PancreasQuiz'
    labels_json = json.loads((raw / 'subtype_labels.json').read_text())
    labels = {case: label for split, mapping in labels_json.items()
              if split != 'validation' for case, label in mapping.items()}
    if args.embedding_layout == 'oof':
        paths = sorted(args.embeddings.glob('fold_*/classification_embeddings/*.npy'))
    else:
        paths = sorted(args.embeddings.glob('classification_embeddings/*.npy'))
    cases = [path.stem for path in paths]
    if len(paths) != 252 or len(set(cases)) != 252 or set(cases) != set(labels):
        raise RuntimeError(f'expected exactly 252 unique OOF embeddings, found {len(paths)}')
    if args.embedding_layout == 'oof':
        x = np.stack([np.load(path)[0] for path in paths]).astype(np.float32)
    else:
        x = np.stack([np.load(path) for path in paths]).astype(np.float32)
    y = np.asarray([labels[case] for case in cases], dtype=np.int64)
    expected_ndim = 2 if args.embedding_layout == 'oof' else 3
    if (x.shape[0] != 252 or x.ndim != expected_ndim
            or (args.embedding_layout != 'oof' and x.shape[1] != 5)
            or not np.isfinite(x).all()):
        raise RuntimeError(f'invalid embedding matrix {x.shape}')
    if args.feature_slice == 'mean':
        x = x[..., :x.shape[-1] // 2]
    elif args.feature_slice == 'max':
        x = x[..., x.shape[-1] // 2:]
    auxiliary_dim = 0
    if args.aux_probabilities is not None:
        auxiliary_mapping = json.loads(args.aux_probabilities.read_text())
        if set(auxiliary_mapping) != set(cases):
            raise RuntimeError('--aux-probabilities do not cover the 252 training cases exactly')
        auxiliary = np.asarray(
            [auxiliary_mapping[case] for case in cases], dtype=np.float32
        )
        if (auxiliary.shape != (len(cases), 3) or not np.isfinite(auxiliary).all()
                or (auxiliary < 0).any()):
            raise RuntimeError(f'invalid auxiliary neural probabilities {auxiliary.shape}')
        auxiliary /= auxiliary.sum(axis=1, keepdims=True).clip(1e-8)
        auxiliary = np.log(auxiliary.clip(1e-6))
        if x.ndim == 3:
            auxiliary = np.repeat(auxiliary[:, None, :], x.shape[1], axis=1)
        x = np.concatenate([x, auxiliary], axis=-1)
        auxiliary_dim = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof = np.zeros((len(y), 3), dtype=np.float32)
    if args.embedding_layout != 'oof' and args.split_mode != 'nnunet':
        raise ValueError('all-fold embeddings require --split-mode nnunet')
    if args.split_mode == 'stratified':
        split_iterator = list(
            StratifiedKFold(5, shuffle=True, random_state=args.seed).split(x, y)
        )
    else:
        split_path = (
            args.primary_repo
            / 'nnUNet_data/nnUNet_preprocessed/Dataset001_PancreasQuiz/splits_final.json'
        )
        nnunet_splits = json.loads(split_path.read_text())
        case_to_index = {case: index for index, case in enumerate(cases)}
        split_iterator = []
        for fold, split in enumerate(nnunet_splits):
            train_idx = np.asarray(
                [case_to_index[case] for case in split['train']], dtype=np.int64
            )
            val_idx = np.asarray(
                [case_to_index[case] for case in split['val']], dtype=np.int64
            )
            if len(set(train_idx) & set(val_idx)) != 0:
                raise RuntimeError(f'nnU-Net fold {fold} has overlapping train and val cases')
            split_iterator.append((train_idx, val_idx))
    run = wandb.init(
        project=os.environ.get('nnUNet_wandb_project', 'uhn-pancreas-quiz'),
        entity=os.environ.get('WANDB_ENTITY') or None,
        group=args.wandb_group,
        name=args.run_name,
        job_type='frozen-embedding-mlp-cv',
        config={
            'input': (
                'encoder_mean_max_plus_oof_cross_attention_log_probabilities'
                if auxiliary_dim else
                'lesion-weighted_frozen_nnunet_bottleneck_mean_max_from_processed_3d_ct'
            ),
            'n_cases': 252, 'input_dim': int(x.shape[-1]), 'hidden_dim': args.hidden_dim,
            'encoder_feature_dim': int(x.shape[-1] - auxiliary_dim),
            'auxiliary_neural_probability_dim': auxiliary_dim,
            'auxiliary_probability_file': (
                str(args.aux_probabilities.resolve()) if args.aux_probabilities else None
            ),
            'auxiliary_probability_transform': 'log' if auxiliary_dim else None,
            'feature_slice': args.feature_slice,
            'split_mode': args.split_mode,
            'embedding_layout': args.embedding_layout,
            'dropout': args.dropout, 'optimizer': 'AdamW', 'lr': args.learning_rate,
            'feature_noise_std': args.feature_noise_std,
            'feature_dropout': args.feature_dropout,
            'weight_decay': args.weight_decay,
            'label_smoothing': args.label_smoothing,
            'class_weight_power': args.class_weight_power,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'min_epochs': args.min_epochs,
            'patience': args.patience,
            'backbone_training': False, 'segmentation_head_training': False,
            'classification_architecture': 'encoder_mean_max_pooling_gelu_mlp',
            'backbone_source': 'same_dataset_original_fivefold_nnunet_checkpoints',
            'external_pretrained_weights': False,
            'external_datasets': False,
            'excluded_inputs': ['morphology', 'volume', 'shape', 'coordinates', 'case_level_tabular_features'],
        },
    )

    for cv_fold, (train_idx, val_idx) in enumerate(split_iterator):
        set_seed(args.seed + cv_fold)
        if args.embedding_layout == 'all-folds-augmented':
            train_features = x[train_idx].reshape(-1, x.shape[-1])
            train_targets = np.repeat(y[train_idx], x.shape[1])
            val_features = x[val_idx, cv_fold, :]
        else:
            fold_x = x if args.embedding_layout == 'oof' else x[:, cv_fold, :]
            train_features = fold_x[train_idx]
            train_targets = y[train_idx]
            val_features = fold_x[val_idx]
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0).clip(1e-5)
        train_x = torch.from_numpy((train_features - mean) / std).to(device)
        val_x = torch.from_numpy((val_features - mean) / std).to(device)
        train_y = torch.from_numpy(train_targets).to(device)
        val_y = torch.from_numpy(y[val_idx]).to(device)
        counts = np.bincount(train_targets, minlength=3)
        class_weights = torch.tensor(
            (len(train_targets) / (3 * counts)) ** args.class_weight_power,
            dtype=torch.float32,
            device=device,
        )
        model = EmbeddingMLP(x.shape[-1], args.hidden_dim, args.dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
        best_f1, best_state, stale = -1.0, None, 0

        for epoch in range(args.max_epochs):
            model.train()
            permutation = torch.randperm(len(train_targets), device=device)
            losses = []
            for start in range(0, len(train_targets), args.batch_size):
                batch = permutation[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                batch_x = train_x[batch]
                if args.feature_dropout > 0:
                    batch_x = F.dropout(batch_x, p=args.feature_dropout, training=True)
                if args.feature_noise_std > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * args.feature_noise_std
                loss = criterion(model(batch_x), train_y[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                probabilities = model(val_x).softmax(1).cpu().numpy()
            metrics = classification_metrics(y[val_idx], probabilities, 3, prefix='')
            f1 = float(metrics['macro_f1'])
            run.log({f'cv_fold_{cv_fold}/train_loss': np.mean(losses),
                     f'cv_fold_{cv_fold}/macro_f1': f1},
                    step=cv_fold * args.max_epochs + epoch)
            if f1 > best_f1 + 1e-6:
                best_f1 = f1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[val_idx] = model(val_x).softmax(1).cpu().numpy()
        torch.save({'state_dict': best_state, 'mean': mean, 'std': std,
                    'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
                    'validation_indices': val_idx, 'cases': cases,
                    'embedding_layout': args.embedding_layout,
                    'embedding_member': cv_fold if args.embedding_layout != 'oof' else None,
                    'auxiliary_neural_probability_dim': auxiliary_dim,
                    'auxiliary_probability_transform': 'log' if auxiliary_dim else None},
                   args.output / 'models' / f'fold_{cv_fold}.pth')
        run.summary[f'cv_fold_{cv_fold}/best_macro_f1'] = best_f1

    predictions = oof.argmax(1)
    metrics = classification_metrics(y, oof, 3, prefix='oof/')
    metrics['oof/confusion_matrix'] = confusion_matrix_counts(y, predictions).tolist()
    metrics['oof/n_cases'] = len(y)
    metrics = {key: value.item() if isinstance(value, np.generic) else value
               for key, value in metrics.items()}
    (args.output / 'summary.json').write_text(json.dumps(metrics, indent=2))
    (args.output / 'oof_probabilities.json').write_text(json.dumps(
        {case: [float(value) for value in probability] for case, probability in zip(cases, oof)}, indent=2))
    run.log({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact(f'{args.run_name}-artifacts', type='frozen-embedding-mlp')
    artifact.add_dir(str(args.output))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
