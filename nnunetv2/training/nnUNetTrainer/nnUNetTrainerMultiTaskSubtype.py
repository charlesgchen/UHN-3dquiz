"""
Multi-task nnU-Net trainer: pancreas/lesion segmentation + per-case lesion subtype classification.

Design notes
------------
Case labels through the dataloader
    nnUNetDataLoader returns {'data', 'target', 'keys'} where 'keys' are the case identifiers, and the
    augmentation transforms never touch 'keys' (see data_loader.py). So the subtype of every sample in
    a batch is a dict lookup on batch['keys'], with no changes to the dataloader itself.

Patch-level training, case-level truth
    Every patch inherits the subtype of the case it was cropped from. That is label noise: a patch that
    happens to miss the lesion carries a subtype label that is not visible in it. We accept it during
    training (the alternative, case-level training, does not fit in memory) and compensate at inference
    by averaging patch predictions over the whole case (see the aggregation in predict_quiz.py).

Class imbalance (train split: 62 / 106 / 84 = subtype 0 / 1 / 2)
    Inverse-frequency class weights in the cross-entropy, computed from the *training* cases only and
    normalised to mean 1 so the effective loss scale (and therefore the seg/cls balance) is unchanged.
    Counting only training cases matters: deriving weights from all 288 labelled cases would leak the
    validation split's class distribution into training.

Overfitting (252 cases, ~7M-parameter head on top of a 3D encoder)
    Four levers, all on by default:
      1. dropout in the classification head (attention dropout + pre-classifier dropout)
      2. label smoothing in the classification CE
      3. the shared encoder is simultaneously constrained by the segmentation loss, which is a strong
         regulariser: the encoder cannot collapse to a 3-way case-level shortcut and still segment
      4. a warmup during which the classification loss is ramped in, so the encoder first learns
         segmentation features rather than fitting 252 case labels from scratch

Model selection
    checkpoint_best continues to track the segmentation pseudo-Dice EMA, exactly as stock nnU-Net.
    The held-out 36-case validation set is never used for selection - only for monitoring and for the
    final report - otherwise the reported held-out numbers would be biased by the selection.
"""

import os
from typing import List, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from torch import autocast, nn

from nnunetv2.evaluation.quiz_metrics import classification_metrics, confusion_matrix_counts, format_report
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.logging.multitask_logger import MultiTaskMetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resenc_unet_with_cls import (
    ResEncUNetWithClassification,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

SUBTYPE_LABELS_FILE = 'subtype_labels.json'


class _SegmentationOnlyView(nn.Module):
    """
    Presents a multi-task network as a segmentation-only network, stashing the classification logits.

    This lets validation_step delegate the whole pseudo-Dice computation to nnUNetTrainer instead of
    copying its ~60 lines (region handling, ignore-label masking, background stripping), which would
    silently drift from upstream. Forward-only, so it is safe under DDP and torch.compile.
    """

    def __init__(self, wrapped: nn.Module, sink: dict):
        super().__init__()
        self.wrapped = wrapped
        self.sink = sink

    # set_deep_supervision_enabled() reaches through to network.decoder, so the view must forward it
    @property
    def decoder(self):
        return self.wrapped.decoder

    @property
    def encoder(self):
        return self.wrapped.encoder

    def forward(self, x):
        segmentation, logits = self.wrapped(x)
        self.sink['logits'] = logits
        return segmentation


class nnUNetTrainerMultiTaskSubtype(nnUNetTrainer):
    # architecture / loss hyperparameters. Kept as class attributes because build_network_architecture
    # is a staticmethod that is also called at inference time, where no trainer instance exists.
    num_subtypes: int = 3
    cls_query_num: int = 4
    cls_num_heads: int = 4
    cls_dropout: float = 0.3
    use_cross_attention: bool = True

    # loss weighting
    cls_loss_weight: float = 0.5
    cls_label_smoothing: float = 0.1
    cls_warmup_epochs: int = 25

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Upgrade (do not rebuild) the logger created by the parent: rebuilding would start a second
        # W&B run and wipe the first one's directory.
        self.logger = MultiTaskMetaLogger.adopt(self.logger)

        self.subtype_labels = self._load_subtype_labels()
        self.cls_class_weights = None  # built in initialize(), needs the training split
        self.cls_loss = None

    # ------------------------------------------------------------------ setup

    def _load_subtype_labels(self) -> dict:
        """case identifier -> subtype, for every labelled case (train and the held-out validation)."""
        path = join(nnUNet_raw, self.plans_manager.dataset_name, SUBTYPE_LABELS_FILE)
        if not isfile(path):
            raise FileNotFoundError(
                f'{SUBTYPE_LABELS_FILE} not found at {path}. It is written by '
                f'nnunetv2/dataset_conversion/Dataset001_PancreasQuiz.py and maps case identifiers to '
                f'the subtype label the classification head is trained on.')
        raw = load_json(path)
        merged = {}
        for split, mapping in raw.items():
            for case_identifier, subtype in mapping.items():
                merged[case_identifier] = int(subtype)
        return merged

    def _compute_class_weights(self, training_identifiers: List[str]) -> torch.Tensor:
        """
        Inverse-frequency weights from the training cases only, normalised to mean 1.

        Normalising to mean 1 keeps the magnitude of the classification loss comparable to the
        unweighted case, so cls_loss_weight keeps meaning the same thing regardless of how skewed the
        split happens to be.
        """
        counts = np.zeros(self.num_subtypes, dtype=np.float64)
        for case_identifier in training_identifiers:
            counts[self.subtype_labels[case_identifier]] += 1
        if (counts == 0).any():
            missing = np.where(counts == 0)[0].tolist()
            raise RuntimeError(f'subtype(s) {missing} have no training cases in this fold; '
                               f'inverse-frequency weighting is undefined')
        weights = counts.sum() / (self.num_subtypes * counts)
        weights = weights / weights.mean()
        self.print_to_log_file(f'subtype counts (train): {counts.astype(int).tolist()}')
        self.print_to_log_file(f'subtype CE weights:     {np.round(weights, 4).tolist()}')
        return torch.from_numpy(weights).float()

    def initialize(self):
        super().initialize()
        training_identifiers, _ = self.do_split()
        self.cls_class_weights = self._compute_class_weights(training_identifiers).to(self.device)
        self.cls_loss = nn.CrossEntropyLoss(weight=self.cls_class_weights,
                                            label_smoothing=self.cls_label_smoothing)
        self.logger.update_config({
            'num_subtypes': self.num_subtypes,
            'cls_query_num': self.cls_query_num,
            'cls_num_heads': self.cls_num_heads,
            'cls_dropout': self.cls_dropout,
            'use_cross_attention': self.use_cross_attention,
            'cls_loss_weight': self.cls_loss_weight,
            'cls_label_smoothing': self.cls_label_smoothing,
            'cls_warmup_epochs': self.cls_warmup_epochs,
            'cls_class_weights': self.cls_class_weights.detach().cpu().numpy().tolist(),
        })

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True) -> nn.Module:
        backbone = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        return ResEncUNetWithClassification(
            backbone,
            num_subtypes=nnUNetTrainerMultiTaskSubtype.num_subtypes,
            query_num=nnUNetTrainerMultiTaskSubtype.cls_query_num,
            num_heads=nnUNetTrainerMultiTaskSubtype.cls_num_heads,
            cls_dropout=nnUNetTrainerMultiTaskSubtype.cls_dropout,
            use_cross_attention=nnUNetTrainerMultiTaskSubtype.use_cross_attention,
        )

    # ------------------------------------------------------------------ helpers

    def _current_cls_weight(self) -> float:
        """Linear warmup of the classification loss weight over the first cls_warmup_epochs."""
        if self.cls_warmup_epochs <= 0:
            return self.cls_loss_weight
        ramp = min(1.0, (self.current_epoch + 1) / self.cls_warmup_epochs)
        return self.cls_loss_weight * ramp

    def _subtype_targets(self, keys) -> torch.Tensor:
        try:
            labels = [self.subtype_labels[k] for k in keys]
        except KeyError as e:
            raise KeyError(f'case {e} has no subtype label in {SUBTYPE_LABELS_FILE}') from e
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ steps

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        cls_target = self._subtype_targets(batch['keys'])

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        cls_weight = self._current_cls_weight()
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_logits = self.network(data)
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.cls_loss(cls_logits.float(), cls_target)
            total_loss = seg_loss + cls_weight * cls_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy(),
                'seg_loss': seg_loss.detach().cpu().numpy(),
                'cls_loss': cls_loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        super().on_train_epoch_end(train_outputs)
        self.logger.log('train_losses_seg', float(np.mean(outputs['seg_loss'])), self.current_epoch)
        self.logger.log('train_losses_cls', float(np.mean(outputs['cls_loss'])), self.current_epoch)
        self.logger.log_metrics_dict({'train/cls_loss_weight': self._current_cls_weight()},
                                     step=self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        cls_target = self._subtype_targets(batch['keys'])

        # Reuse the parent's segmentation validation logic verbatim by temporarily presenting a
        # segmentation-only view of the network. This keeps the pseudo-Dice computation identical to
        # stock nnU-Net instead of duplicating it here (where it would drift out of sync).
        network = self.network
        cls_logits_holder = {}
        self.network = _SegmentationOnlyView(network, cls_logits_holder)
        try:
            out = super().validation_step(batch)
        finally:
            self.network = network

        cls_logits = cls_logits_holder['logits'].float()
        cls_loss = self.cls_loss(cls_logits, cls_target)
        # the parent's 'loss' is the segmentation loss only; report both parts and the combined value
        out['seg_loss'] = out['loss']
        out['cls_loss'] = cls_loss.detach().cpu().numpy()
        out['loss'] = out['loss'] + self._current_cls_weight() * out['cls_loss']
        out['cls_probs'] = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
        out['cls_target'] = cls_target.detach().cpu().numpy()
        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)

        self.logger.log('val_losses_seg', float(np.mean(outputs['seg_loss'])), self.current_epoch)
        self.logger.log('val_losses_cls', float(np.mean(outputs['cls_loss'])), self.current_epoch)

        probs = np.concatenate([np.atleast_2d(p) for p in outputs['cls_probs']], axis=0)
        targets = np.concatenate([np.atleast_1d(t) for t in outputs['cls_target']], axis=0)

        metrics = classification_metrics(targets, probs, num_classes=self.num_subtypes,
                                         prefix='val_patch_cls/')
        # headline metrics also go to the local log so they appear in progress.png and survive resume
        self.logger.log('cls_balanced_accuracy', metrics['val_patch_cls/balanced_accuracy'], self.current_epoch)
        self.logger.log('cls_macro_f1', metrics['val_patch_cls/macro_f1'], self.current_epoch)
        self.logger.log('cls_mcc', metrics['val_patch_cls/mcc'], self.current_epoch)
        self.logger.log_metrics_dict(metrics, step=self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        End-of-training sliding-window validation on the fold's internal validation split.

        nnUNetPredictor calls self.network(x) and expects a tensor, so the multi-task network is
        presented through the segmentation-only view for the duration. Classification on whole cases
        is handled separately by nnunetv2/inference/predict_quiz.py, which aggregates patch
        predictions per case.
        """
        network = self.network
        self.network = _SegmentationOnlyView(network, {})
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            self.network = network

    def on_epoch_end(self):
        super().on_epoch_end()
        self.print_to_log_file(
            'cls (patch-level, internal val): '
            f"bal_acc {np.round(self.logger.get_value('cls_balanced_accuracy', -1), 4)}, "
            f"macro_F1 {np.round(self.logger.get_value('cls_macro_f1', -1), 4)}, "
            f"MCC {np.round(self.logger.get_value('cls_mcc', -1), 4)}")
