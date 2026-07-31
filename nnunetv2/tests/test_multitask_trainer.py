"""
Smoke tests for nnUNetTrainerMultiTaskSubtype.

These run the real trainer on the real preprocessed dataset, on CPU, with a shrunken patch size and a
handful of iterations. They are slow (minutes) but they exercise the parts most likely to break:
subtype lookup through the dataloader, the joint loss, output collation, and metric logging.

Skipped unless Dataset001_PancreasQuiz has been preprocessed.
"""

import json
import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTaskSubtype import nnUNetTrainerMultiTaskSubtype

DATASET = 'Dataset001_PancreasQuiz'
PREPROCESSED = os.environ.get('nnUNet_preprocessed', '')
PLANS_PATH = os.path.join(PREPROCESSED, DATASET, 'nnUNetResEncUNetMPlans.json')
DATASET_JSON_PATH = os.path.join(PREPROCESSED, DATASET, 'dataset.json')

needs_data = pytest.mark.skipif(
    not (os.path.isfile(PLANS_PATH) and os.path.isfile(DATASET_JSON_PATH)),
    reason='Dataset001_PancreasQuiz not preprocessed; run nnUNetv2_plan_and_preprocess first')


def _tiny_trainer(tmp_path, monkeypatch, patch_size=(32, 64, 64)):
    """Trainer on CPU with a small patch size and very few iterations."""
    monkeypatch.setenv('nnUNet_wandb_enabled', '0')
    monkeypatch.setenv('nnUNet_results', str(tmp_path))

    with open(PLANS_PATH) as f:
        plans = json.load(f)
    with open(DATASET_JSON_PATH) as f:
        dataset_json = json.load(f)

    plans = deepcopy(plans)
    plans['configurations']['3d_fullres']['patch_size'] = list(patch_size)
    plans['configurations']['3d_fullres']['batch_size'] = 2
    plans['continue_training'] = False

    # nnUNet_raw / nnUNet_preprocessed / nnUNet_results are _EnvPath objects that read os.environ
    # lazily, so monkeypatch.setenv above is enough to redirect results into tmp_path. Do NOT
    # setattr them to plain strings: the trainer calls nnUNet_results.is_set().
    trainer = nnUNetTrainerMultiTaskSubtype(plans, '3d_fullres', 0, dataset_json,
                                            device=torch.device('cpu'))
    trainer.num_epochs = 2
    trainer.num_iterations_per_epoch = 2
    trainer.num_val_iterations_per_epoch = 2
    trainer.save_every = 100  # don't write checkpoints during the smoke test
    return trainer


# ------------------------------------------------------------------ label plumbing (fast, no model)

@needs_data
def test_subtype_labels_cover_all_cases(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    assert len(trainer.subtype_labels) == 288, 'expected 252 train + 36 validation cases'
    assert set(trainer.subtype_labels.values()) == {0, 1, 2}


@needs_data
def test_class_weights_are_inverse_frequency_and_mean_one(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    train_ids = list(trainer.subtype_labels)
    counts = np.zeros(3)
    for k in train_ids:
        counts[trainer.subtype_labels[k]] += 1

    weights = trainer._compute_class_weights(train_ids).numpy()
    assert np.isclose(weights.mean(), 1.0), 'weights must be normalised to mean 1'
    # rarest class must get the largest weight
    assert np.argmax(weights) == np.argmin(counts)
    assert np.argmin(weights) == np.argmax(counts)


@needs_data
def test_class_weights_use_only_the_given_identifiers(tmp_path, monkeypatch):
    """Guards the leak: weights must come from training cases, not from all 288 labelled cases."""
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    only_two_classes = [k for k, v in trainer.subtype_labels.items() if v in (0, 1)]
    with pytest.raises(RuntimeError, match='no training cases'):
        trainer._compute_class_weights(only_two_classes)


@needs_data
def test_warmup_ramps_classification_weight(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.cls_warmup_epochs = 10
    trainer.current_epoch = 0
    first = trainer._current_cls_weight()
    trainer.current_epoch = 4
    mid = trainer._current_cls_weight()
    trainer.current_epoch = 50
    late = trainer._current_cls_weight()

    assert 0 < first < mid < late
    assert np.isclose(late, trainer.cls_loss_weight), 'weight must saturate at cls_loss_weight'


@needs_data
def test_unknown_case_identifier_raises_clearly(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.device = torch.device('cpu')
    with pytest.raises(KeyError, match='no subtype label'):
        trainer._subtype_targets(['not_a_real_case'])


# ------------------------------------------------------------------ full training loop (slow)

@needs_data
@pytest.mark.slow
def test_two_epochs_run_and_log_both_tasks(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.run_training()

    for key in ('train_losses', 'val_losses', 'train_losses_seg', 'train_losses_cls',
                'val_losses_seg', 'val_losses_cls', 'cls_balanced_accuracy', 'cls_macro_f1', 'cls_mcc'):
        values = trainer.logger.get_value(key, step=None)
        assert len(values) == 2, f'{key} should have one value per epoch, got {len(values)}'
        assert all(np.isfinite(v) for v in values), f'{key} contains non-finite values: {values}'

    assert os.path.isfile(os.path.join(trainer.output_folder, 'progress.png'))


@needs_data
@pytest.mark.slow
def test_network_is_multitask_and_losses_are_separate(tmp_path, monkeypatch):
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resenc_unet_with_cls import (
        ResEncUNetWithClassification,
    )
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    # dataloaders are created in on_train_start (nnUNetTrainer.py:946), not in initialize()
    trainer.on_train_start()
    assert isinstance(trainer.network, ResEncUNetWithClassification)

    batch = next(trainer.dataloader_train)
    out = trainer.train_step(batch)
    assert {'loss', 'seg_loss', 'cls_loss'} <= set(out)
    assert np.isfinite(out['seg_loss']) and np.isfinite(out['cls_loss'])
    # total = seg + warmed-up weight * cls
    expected = out['seg_loss'] + trainer._current_cls_weight() * out['cls_loss']
    assert np.isclose(float(out['loss']), float(expected), rtol=1e-4)


@needs_data
@pytest.mark.slow
def test_validation_step_returns_probabilities_and_targets(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    # dataloaders are created in on_train_start (nnUNetTrainer.py:946), not in initialize()
    trainer.on_train_start()
    trainer.on_validation_epoch_start()

    batch = next(trainer.dataloader_val)
    out = trainer.validation_step(batch)

    assert out['cls_probs'].shape == (trainer.batch_size, 3)
    assert np.allclose(out['cls_probs'].sum(axis=1), 1.0, atol=1e-4), 'probabilities must be normalised'
    assert out['cls_target'].shape == (trainer.batch_size,)
    assert set(np.unique(out['cls_target'])) <= {0, 1, 2}
    # pseudo-dice bookkeeping from the parent must survive the seg-only view
    assert {'tp_hard', 'fp_hard', 'fn_hard'} <= set(out)
