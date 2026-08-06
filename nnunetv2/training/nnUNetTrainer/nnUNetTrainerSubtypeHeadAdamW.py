"""Second-stage subtype-head training on a fold-matched segmentation checkpoint.

The joint SGD run is deliberately retained as the from-scratch segmentation stage.  This trainer is
the classification stage: it loads that fold's checkpoint, freezes the encoder and decoder, resets the
collapsed classification head in place, and optimizes only the head with AdamW at 1e-4.  Each saved
checkpoint still contains the complete shared-encoder multi-task model, so the usual quiz predictor can
produce both segmentation and subtype outputs without stitching together separate networks.

This follows the segmentation-first/ROI-classifier pattern used by the strongest alternative-plan
solution while preserving the project's required ResEnc-M shared encoder and five-fold workflow.
"""

import os
from typing import List

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTaskSubtypeStable import (
    nnUNetTrainerMultiTaskSubtypeStable,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerSubtypeHeadAdamW(nnUNetTrainerMultiTaskSubtypeStable):
    """Freeze segmentation features and fit a freshly reset subtype head with AdamW."""

    cls_loss_weight: float = 1.0
    cls_warmup_epochs: int = 0
    cls_label_smoothing: float = 0.05

    convergence_metric: str = 'cls_macro_f1'
    convergence_patience: int = 15
    convergence_min_epochs: int = 15
    convergence_smoothing: float = 0.8
    convergence_anneal_epochs: int = 10

    head_initial_lr: float = 1e-4
    head_weight_decay: float = 1e-4
    head_num_epochs: int = 75
    head_grad_clip: float = 1.0
    source_checkpoint_env: str = 'nnUNet_subtype_source_checkpoint'

    # checkpoint_best remains nnU-Net's segmentation-selected checkpoint. This independent checkpoint
    # follows a smoothed INTERNAL fold-validation macro-F1 so classifier selection never consults the
    # 36 held-out cases.
    cls_selection_smoothing: float = 0.8
    cls_selection_min_epochs: int = 10
    cls_selection_min_delta: float = 1e-3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Keep this signature explicit: nnUNetTrainer introspects constructor locals to serialize the
        # init arguments into every checkpoint, so a generic *args/**kwargs wrapper is not compatible.
        super().__init__(plans, configuration, fold, dataset_json, device)
        # nnUNetTrainer initializes these as instance attributes, so set the phase-specific values
        # after its constructor has run.
        self.initial_lr = self.head_initial_lr
        self.weight_decay = self.head_weight_decay
        self.num_epochs = self.head_num_epochs
        self.oversample_foreground_percent = 0.66
        self._head_was_reset = False
        self._source_was_restored = False
        self._cls_selection_ema = None
        self._best_cls_selection_ema = -float('inf')
        self._best_cls_selection_epoch = None

    def _do_i_compile(self):
        # The training step intentionally invokes encoder and head separately. Compiling the wrapper's
        # unused joint forward only adds startup cost and obscures access to those two modules.
        return False

    def _raw_network(self) -> nn.Module:
        network = self.network.module if self.is_ddp else self.network
        return network._orig_mod if hasattr(network, '_orig_mod') else network

    def configure_optimizers(self):
        network = self._raw_network()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        for parameter in network.classification_head.parameters():
            parameter.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            network.classification_head.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def initialize(self):
        super().initialize()
        network = self._raw_network()
        trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in network.parameters())
        self.print_to_log_file(
            f'classification-head phase: AdamW lr={self.initial_lr}, weight_decay={self.weight_decay}; '
            f'trainable parameters {trainable:,}/{total:,}')
        self.logger.update_config({
            'training_phase': 'fold_matched_segmentation_checkpoint_to_subtype_head',
            'optimizer': 'AdamW',
            'head_initial_lr': self.initial_lr,
            'head_weight_decay': self.weight_decay,
            'head_trainable_parameters': trainable,
            'frozen_parameters': total - trainable,
            'head_reset_before_training': True,
            'source_checkpoint': os.environ.get(self.source_checkpoint_env),
            'cls_selection_smoothing': self.cls_selection_smoothing,
            'cls_selection_min_epochs': self.cls_selection_min_epochs,
            'cls_selection_min_delta': self.cls_selection_min_delta,
        })

    def _restore_fold_matched_checkpoint(self):
        """Restore every parameter, including seg heads skipped by nnU-Net's generic transfer loader."""
        source = os.environ.get(self.source_checkpoint_env)
        if not source:
            raise RuntimeError(
                f'{self.source_checkpoint_env} must point to this fold\'s completed segmentation checkpoint. '
                'The generic -pretrained_weights loader is insufficient because it intentionally skips '
                'segmentation output layers.')
        if not os.path.isfile(source):
            raise FileNotFoundError(f'{self.source_checkpoint_env} does not exist: {source}')

        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        source_fold = checkpoint.get('init_args', {}).get('fold')
        if source_fold is not None and int(source_fold) != int(self.fold):
            raise RuntimeError(
                f'fold mismatch: trainer fold {self.fold} cannot use fold {source_fold} checkpoint {source}')
        self._raw_network().load_state_dict(checkpoint['network_weights'], strict=True)
        self._source_was_restored = True

    def _reset_classification_head(self):
        """Reset every head parameter in place so optimizer parameter references remain valid."""
        head = self._raw_network().classification_head
        for module in head.modules():
            if module is not head and hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        pooling = head.pooling
        # MultiheadAttention owns its input projection weights directly rather than in a child Linear.
        if hasattr(pooling, 'cross_attention'):
            pooling.cross_attention._reset_parameters()
        # CrossAttentionPooling also owns the learned query directly.
        if hasattr(pooling, '_init_weights'):
            pooling._init_weights()
        self._head_was_reset = True

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        # Restore explicitly instead of relying on nnU-Net's generic transfer loader: that helper
        # deliberately omits segmentation output layers. Then reset only the unstable first-stage head.
        if self.current_epoch == 0 and not self._head_was_reset:
            self._restore_fold_matched_checkpoint()
            self._reset_classification_head()
            self.print_to_log_file(
                f'Restored full fold-matched checkpoint from {os.environ[self.source_checkpoint_env]}; '
                'reset classification head only.')
        super().on_train_start()
        if self.current_epoch > 0:
            self._rebuild_cls_selection_state()

    @property
    def cls_best_checkpoint_file(self):
        return os.path.join(self.output_folder, 'checkpoint_best_cls.pth')

    def _update_cls_selection_state(self, value: float, epoch: int, save: bool):
        if self._cls_selection_ema is None:
            self._cls_selection_ema = float(value)
        else:
            smoothing = float(self.cls_selection_smoothing)
            self._cls_selection_ema = smoothing * self._cls_selection_ema + (1.0 - smoothing) * float(value)

        improved = (epoch >= self.cls_selection_min_epochs and
                    self._cls_selection_ema > self._best_cls_selection_ema + self.cls_selection_min_delta)
        if improved:
            self._best_cls_selection_ema = self._cls_selection_ema
            self._best_cls_selection_epoch = int(epoch)
            if save:
                self.print_to_log_file(
                    f'New best internal classification EMA: {self._best_cls_selection_ema:.4f} '
                    f'at epoch {epoch}; saving checkpoint_best_cls.pth')
                self.save_checkpoint(self.cls_best_checkpoint_file)
                self.logger.log_summary('classification_selection/best_ema', self._best_cls_selection_ema)
                self.logger.log_summary('classification_selection/best_epoch', self._best_cls_selection_epoch)

    def _rebuild_cls_selection_state(self):
        """Recover classification-selection EMA from logger history after --c resume."""
        self._cls_selection_ema = None
        self._best_cls_selection_ema = -float('inf')
        self._best_cls_selection_epoch = None
        values = self.logger.get_value('cls_macro_f1', step=None)
        for epoch, value in enumerate(values[:self.current_epoch]):
            self._update_cls_selection_state(float(value), epoch, save=False)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        value = float(self.logger.get_value('cls_macro_f1', step=-1))
        if np.isfinite(value):
            # Called before on_epoch_end increments current_epoch, so save_checkpoint records the same
            # completed-epoch convention as nnU-Net's native best/latest checkpoints.
            self._update_cls_selection_state(value, self.current_epoch, save=True)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        network = self._raw_network()
        network.backbone.eval()
        network.classification_head.train()

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'][0] if isinstance(batch['target'], list) else batch['target']
        target = target.to(self.device, non_blocking=True)
        cls_target = self._subtype_targets(batch['keys'])
        patch_weights = self._patch_weights(target)

        self.optimizer.zero_grad(set_to_none=True)
        network = self._raw_network()
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context()
        with amp_context:
            with torch.no_grad():
                bottleneck = network.backbone.encoder(data)[-1]
            cls_logits = network.classification_head(bottleneck.detach())
            cls_loss = self._weighted_cls_loss(cls_logits.float(), cls_target, patch_weights)
            total_loss = self._current_cls_weight() * cls_loss

        head_parameters = network.classification_head.parameters()
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(head_parameters, self.head_grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, self.head_grad_clip)
            self.optimizer.step()

        # Preserve the multi-task logger's schema. Segmentation is frozen, so its train loss is exactly
        # zero in this phase; validation still measures the retained decoder normally.
        zero = np.asarray(0.0, dtype=np.float32)
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': zero,
            'cls_loss': cls_loss.detach().cpu().numpy(),
            'cls_patch_weight': float(patch_weights.mean()),
        }
