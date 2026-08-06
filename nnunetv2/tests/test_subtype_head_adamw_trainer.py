import inspect

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSubtypeHeadAdamW import (
    nnUNetTrainerSubtypeHeadAdamW,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resenc_unet_with_cls import (
    ClassificationHead,
)


class _TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.classification_head = ClassificationHead(
            embed_dim=4,
            num_classes=3,
            query_num=2,
            dropout=0.0,
            use_cross_attention=True,
            num_heads=2,
        )


def _trainer_shell():
    trainer = object.__new__(nnUNetTrainerSubtypeHeadAdamW)
    trainer.network = _TinyNetwork()
    trainer.is_ddp = False
    trainer.initial_lr = 1e-4
    trainer.weight_decay = 1e-4
    trainer.num_epochs = 75
    trainer.device = torch.device('cpu')
    trainer.fold = 0
    return trainer


def test_optimizer_updates_only_classification_head():
    trainer = _trainer_shell()
    optimizer, _ = trainer.configure_optimizers()

    assert all(not parameter.requires_grad for parameter in trainer.network.backbone.parameters())
    assert all(parameter.requires_grad for parameter in trainer.network.classification_head.parameters())
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    expected = {id(parameter) for parameter in trainer.network.classification_head.parameters()}
    assert optimized == expected
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['lr'] == trainer.initial_lr


def test_head_reset_is_in_place_and_changes_collapsed_parameters():
    trainer = _trainer_shell()
    optimizer, _ = trainer.configure_optimizers()
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}

    with torch.no_grad():
        for parameter in trainer.network.classification_head.parameters():
            parameter.fill_(7.0)
    trainer._head_was_reset = False
    trainer._reset_classification_head()

    reset_parameter_ids = {id(parameter) for parameter in trainer.network.classification_head.parameters()}
    assert parameter_ids == reset_parameter_ids, 'reset must not invalidate optimizer parameter references'
    assert trainer._head_was_reset
    assert any(not torch.all(parameter == 7.0) for parameter in trainer.network.classification_head.parameters())


def test_phase_hyperparameters_target_classification_convergence():
    assert nnUNetTrainerSubtypeHeadAdamW.cls_loss_weight == 1.0
    assert nnUNetTrainerSubtypeHeadAdamW.cls_warmup_epochs == 0
    assert nnUNetTrainerSubtypeHeadAdamW.convergence_metric == 'cls_macro_f1'
    assert nnUNetTrainerSubtypeHeadAdamW.head_initial_lr == 1e-4


def test_constructor_keeps_nnunet_checkpoint_signature():
    parameters = inspect.signature(nnUNetTrainerSubtypeHeadAdamW.__init__).parameters
    assert tuple(parameters) == ('self', 'plans', 'configuration', 'fold', 'dataset_json', 'device')


def test_full_fold_checkpoint_restore_includes_segmentation_parameters(tmp_path, monkeypatch):
    source_network = _TinyNetwork()
    with torch.no_grad():
        for parameter in source_network.backbone.parameters():
            parameter.fill_(3.0)
        for parameter in source_network.classification_head.parameters():
            parameter.fill_(7.0)

    checkpoint = tmp_path / 'fold0.pth'
    torch.save({
        'network_weights': source_network.state_dict(),
        'init_args': {'fold': 0},
    }, checkpoint)
    monkeypatch.setenv(nnUNetTrainerSubtypeHeadAdamW.source_checkpoint_env, str(checkpoint))

    trainer = _trainer_shell()
    trainer._source_was_restored = False
    trainer._restore_fold_matched_checkpoint()

    assert trainer._source_was_restored
    assert all(torch.all(parameter == 3.0) for parameter in trainer.network.backbone.parameters())
    assert all(torch.all(parameter == 7.0) for parameter in trainer.network.classification_head.parameters())


def test_fold_checkpoint_restore_rejects_cross_fold_source(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'fold1.pth'
    torch.save({
        'network_weights': _TinyNetwork().state_dict(),
        'init_args': {'fold': 1},
    }, checkpoint)
    monkeypatch.setenv(nnUNetTrainerSubtypeHeadAdamW.source_checkpoint_env, str(checkpoint))

    trainer = _trainer_shell()
    try:
        trainer._restore_fold_matched_checkpoint()
    except RuntimeError as error:
        assert 'fold mismatch' in str(error)
    else:
        raise AssertionError('cross-fold source checkpoint must be rejected')


def test_classification_checkpoint_selection_uses_smoothed_internal_metric(tmp_path):
    trainer = _trainer_shell()
    trainer.output_folder = str(tmp_path)
    trainer.cls_selection_smoothing = 0.5
    trainer.cls_selection_min_epochs = 2
    trainer.cls_selection_min_delta = 0.01
    trainer._cls_selection_ema = None
    trainer._best_cls_selection_ema = -float('inf')
    trainer._best_cls_selection_epoch = None
    saved = []
    trainer.print_to_log_file = lambda *args, **kwargs: None
    trainer.save_checkpoint = lambda path: saved.append(path)

    class _Logger:
        def log_summary(self, *args, **kwargs):
            pass

    trainer.logger = _Logger()
    for epoch, value in enumerate((0.2, 0.4, 0.5, 0.3, 0.7)):
        trainer._update_cls_selection_state(value, epoch, save=True)

    assert len(saved) == 2
    assert saved[-1].endswith('checkpoint_best_cls.pth')
    assert trainer._best_cls_selection_epoch == 4
    assert abs(trainer._best_cls_selection_ema - 0.525) < 1e-8
