import os

import numpy as np
import pytest

from nnunetv2.evaluation.quiz_metrics import classification_metrics
from nnunetv2.training.logging.multitask_logger import (
    CORE_KEYS,
    MULTITASK_KEYS,
    MultiTaskLocalLogger,
    MultiTaskMetaLogger,
)


def _log_one_epoch(logger, epoch, cls_metrics=True):
    for key in CORE_KEYS:
        if key == 'dice_per_class_or_region':
            logger.log(key, [0.9, 0.3], epoch)
        elif key == 'ema_fg_dice':
            continue  # derived automatically from mean_fg_dice by MetaLogger
        else:
            logger.log(key, 0.5 + 0.01 * epoch, epoch)
    for key in MULTITASK_KEYS:
        if not cls_metrics and key.startswith('cls_'):
            continue
        logger.log(key, 0.4 + 0.01 * epoch, epoch)


def test_local_logger_has_all_keys():
    logger = MultiTaskLocalLogger()
    for key in CORE_KEYS + MULTITASK_KEYS:
        assert key in logger.my_fantastic_logging


def test_checkpoint_roundtrip_and_backward_compatibility():
    """A checkpoint written by the stock trainer has no multi-task keys; resuming must not crash."""
    logger = MultiTaskLocalLogger()
    legacy_checkpoint = {key: [0.1, 0.2] for key in CORE_KEYS}
    logger.load_checkpoint(legacy_checkpoint)
    for key in MULTITASK_KEYS:
        assert logger.my_fantastic_logging[key] == []
    assert logger.my_fantastic_logging['train_losses'] == [0.1, 0.2]


def test_progress_png_written(tmp_path):
    logger = MultiTaskLocalLogger()
    for epoch in range(4):
        for key in CORE_KEYS:
            logger.log(key, [0.9, 0.3] if key == 'dice_per_class_or_region' else 0.5, epoch)
        for key in MULTITASK_KEYS:
            logger.log(key, 0.4, epoch)
    logger.plot_progress_png(str(tmp_path))
    out = tmp_path / 'progress.png'
    assert out.is_file() and out.stat().st_size > 0


def test_progress_png_survives_missing_multitask_values(tmp_path):
    """Classification metrics may legitimately lag the core keys; plotting must still work."""
    logger = MultiTaskLocalLogger()
    for epoch in range(3):
        for key in CORE_KEYS:
            logger.log(key, [0.9, 0.3] if key == 'dice_per_class_or_region' else 0.5, epoch)
    logger.plot_progress_png(str(tmp_path))
    assert (tmp_path / 'progress.png').is_file()


def test_progress_png_noop_before_first_epoch(tmp_path):
    MultiTaskLocalLogger().plot_progress_png(str(tmp_path))
    assert not (tmp_path / 'progress.png').exists()


def test_meta_logger_offline_end_to_end(tmp_path, monkeypatch):
    """Exercise the real W&B code path in offline mode: no network, no credentials needed."""
    pytest.importorskip('wandb')
    monkeypatch.setenv('nnUNet_wandb_enabled', '1')
    monkeypatch.setenv('nnUNet_wandb_mode', 'offline')
    monkeypatch.setenv('nnUNet_wandb_project', 'pytest-quiz')
    monkeypatch.setenv('WANDB_SILENT', 'true')

    logger = MultiTaskMetaLogger(str(tmp_path), resume=False)
    assert isinstance(logger.local_logger, MultiTaskLocalLogger)
    assert len(logger.loggers) == 1, 'W&B logger should be attached when nnUNet_wandb_enabled=1'

    logger.update_config({'trainer': 'test', 'fold': 0})

    rng = np.random.default_rng(0)
    for epoch in range(3):
        _log_one_epoch(logger, epoch)
        y_true = np.concatenate([np.full(9, 0), np.full(15, 1), np.full(12, 2)])
        y_prob = rng.dirichlet(np.ones(3), size=len(y_true))
        logger.log_metrics_dict(classification_metrics(y_true, y_prob), step=epoch)

    logger.log_summary('final_val/cls_macro_f1', 0.71)
    logger.plot_progress_png(str(tmp_path))

    assert (tmp_path / 'progress.png').is_file()
    assert (tmp_path / 'wandb').is_dir(), 'offline run directory should exist'
    # local log still holds exactly one value per epoch for every core key
    for key in CORE_KEYS:
        assert len(logger.get_value(key, step=None)) == 3, key


def test_log_metrics_dict_tolerates_nan_and_none(tmp_path, monkeypatch):
    pytest.importorskip('wandb')
    monkeypatch.setenv('nnUNet_wandb_enabled', '1')
    monkeypatch.setenv('nnUNet_wandb_mode', 'offline')
    monkeypatch.setenv('WANDB_SILENT', 'true')

    logger = MultiTaskMetaLogger(str(tmp_path), resume=False)
    logger.log_metrics_dict({'cls/auroc_macro_ovr': float('nan'), 'cls/skipped': None, 'cls/mcc': 0.5}, step=0)


def test_meta_logger_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv('nnUNet_wandb_enabled', raising=False)
    logger = MultiTaskMetaLogger(str(tmp_path), resume=False)
    assert logger.loggers == [], 'W&B must stay off unless explicitly enabled'
    logger.log_metrics_dict({'cls/mcc': 0.5}, step=0)  # must be a silent no-op
