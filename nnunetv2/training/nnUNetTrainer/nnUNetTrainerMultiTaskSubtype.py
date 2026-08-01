"""
Multi-task nnU-Net trainer: pancreas/lesion segmentation + per-case lesion subtype classification.

Design notes
------------
Case labels through the dataloader
    nnUNetDataLoader returns {'data', 'target', 'keys'} where 'keys' are the case identifiers, and the
    augmentation transforms never touch 'keys' (see data_loader.py). So the subtype of every sample in
    a batch is a dict lookup on batch['keys'], with no changes to the dataloader itself.

Patch-level training, case-level truth
    Every patch inherits the subtype of the case it was cropped from, but that label is only *visible*
    in patches that actually contain pancreatic tissue. Case-level training does not fit in memory, so
    the mismatch has to be handled in the loss instead. `_patch_weights` grades each patch:

        no foreground at all     -> weight 0
            pure background or neighbouring organs. The subtype is not inferable from such a patch, so
            the target is pure label noise and contributes nothing but gradient variance.
        pancreas, but no lesion  -> weight cls_patch_weight_floor
            the lesion itself is absent, but parenchymal texture, atrophy and duct calibre do carry
            subtype signal, so these patches are down-weighted rather than dropped.
        lesion present           -> weight 1

    Set cls_patch_weighting=False to recover the unweighted baseline for an ablation.

    The same idea is applied at inference with the *predicted* segmentation: predict_quiz.py weights
    each patch's subtype probabilities by how much foreground the network sees in it before averaging
    them into one prediction per case.

Reported classification metrics
    The quiz scores one subtype per *case*, so on_validation_epoch_end aggregates the internal split's
    patch probabilities by case and reports both levels. The case-level numbers are the ones that drive
    convergence detection; patch-level macro-F1 is kept as a diagnostic but systematically understates
    performance, because it counts patches whose label is not inferable in the first place.

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
from typing import List, Optional, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json
from torch import autocast, nn

from nnunetv2.evaluation.quiz_metrics import classification_metrics, confusion_matrix_counts, format_report
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.convergence import (
    AnnealOutLRScheduler,
    ConvergenceDetector,
    ConvergenceReached,
)
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

    # per-patch weighting of the classification loss (see the module docstring). The floor is the
    # weight given to a patch that contains pancreas but no lesion; 1.0 disables the down-weighting
    # while still dropping patches with no foreground at all.
    cls_patch_weighting: bool = True
    cls_patch_weight_floor: float = 0.3

    # convergence detection / early stopping. See nnunetv2/training/convergence.py for why stopping
    # is followed by an anneal-out phase rather than being immediate.
    enable_early_stopping: bool = True
    convergence_metric: str = 'combined'      # 'combined' | 'ema_fg_dice' | 'cls_macro_f1'
    convergence_patience: int = 50
    convergence_min_delta: float = 1e-3
    convergence_min_epochs: int = 100
    convergence_smoothing: float = 0.9
    convergence_anneal_epochs: int = 25

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Upgrade (do not rebuild) the logger created by the parent: rebuilding would start a second
        # W&B run and wipe the first one's directory.
        self.logger = MultiTaskMetaLogger.adopt(self.logger)

        self.subtype_labels = self._load_subtype_labels()
        self.lesion_label = self._resolve_lesion_label()
        self.cls_class_weights = None  # built in initialize(), needs the training split
        self.cls_loss = None

        self.convergence_detector = ConvergenceDetector(
            patience=self.convergence_patience,
            min_delta=self.convergence_min_delta,
            min_epochs=max(self.convergence_min_epochs, self.cls_warmup_epochs),
            smoothing=self.convergence_smoothing,
        )
        # all set when the anneal-out phase starts
        self._final_epoch: Optional[int] = None
        self._anneal_start_epoch: Optional[int] = None
        self._anneal_start_lr: Optional[float] = None

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

    def _resolve_lesion_label(self) -> int:
        """
        The integer label of the lesion, used to grade patches in _patch_weights.

        Prefers a label literally named 'lesion' (what Dataset001_PancreasQuiz writes) and otherwise
        falls back to the highest foreground label, which is the convention for a nested structure.
        """
        labels = self.dataset_json.get('labels', {}) if self.dataset_json else {}
        for name, value in labels.items():
            if str(name).lower() == 'lesion' and isinstance(value, int):
                return int(value)
        foreground = [int(v) for v in labels.values() if isinstance(v, int) and v > 0]
        if not foreground:
            raise RuntimeError(
                f"cannot determine the lesion label from dataset.json labels {labels!r}; the "
                f"classification patch weighting needs to know which label marks the lesion")
        return max(foreground)

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
        # reduction='none' because _weighted_cls_loss applies the per-patch weights and does the
        # reduction itself; see there for why the denominator is reproduced by hand
        self.cls_loss = nn.CrossEntropyLoss(weight=self.cls_class_weights,
                                            label_smoothing=self.cls_label_smoothing,
                                            reduction='none')
        if self.cls_patch_weighting:
            self.print_to_log_file(
                f'classification patch weighting: lesion label {self.lesion_label} -> weight 1.0, '
                f'foreground without lesion -> {self.cls_patch_weight_floor}, no foreground -> 0.0')
        else:
            self.print_to_log_file('classification patch weighting: disabled (every patch weight 1.0)')
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
            'cls_patch_weighting': self.cls_patch_weighting,
            'cls_patch_weight_floor': self.cls_patch_weight_floor,
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

    def _patch_weights(self, seg_target: torch.Tensor) -> torch.Tensor:
        """
        Per-patch weight for the classification loss, in [0, 1]. See the module docstring for the
        three tiers and why they are graded this way.

        seg_target is the full-resolution ground truth, [B, 1, X, Y, Z]. nnU-Net writes its ignore
        label as a negative value, so foreground is `> 0` rather than `!= 0`.
        """
        flat = seg_target.flatten(1)
        if not self.cls_patch_weighting:
            return torch.ones(flat.shape[0], dtype=torch.float32, device=flat.device)
        has_foreground = (flat > 0).any(dim=1).float()
        has_lesion = (flat == self.lesion_label).any(dim=1).float()
        floor = float(self.cls_patch_weight_floor)
        # lesion -> floor + (1 - floor) = 1, foreground only -> floor, nothing -> 0
        return (floor + (1.0 - floor) * has_lesion) * has_foreground

    def _weighted_cls_loss(self, cls_logits: torch.Tensor, cls_target: torch.Tensor,
                           patch_weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted mean CE that preserves the scale of CrossEntropyLoss(weight=..., reduction='mean').

        PyTorch's weighted 'mean' reduction divides by the sum of the *class* weights of the targets,
        not by the batch size. Reproducing that denominator with the patch weights folded in is what
        keeps cls_loss_weight meaning the same thing whether patch weighting is on or off - normalising
        by the batch size instead would silently rescale the seg/cls balance the moment weighting was
        enabled, and every cls_loss_weight tuned without it would become meaningless.

        With uniform patch weights this is numerically identical to the previous reduction='mean'.
        """
        per_sample = self.cls_loss(cls_logits, cls_target)   # already scaled by the class weights
        denominator = (patch_weights * self.cls_class_weights[cls_target]).sum()
        if float(denominator) <= 1e-8:
            # no patch in this batch carries any evidence for its subtype: contribute no gradient
            # rather than dividing by ~0. Kept in the graph so AMP and DDP see a real tensor.
            return per_sample.sum() * 0.0
        return (patch_weights * per_sample).sum() / denominator

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
        patch_weights = self._patch_weights(target[0] if isinstance(target, list) else target)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, cls_logits = self.network(data)
            seg_loss = self.loss(seg_output, target)
            cls_loss = self._weighted_cls_loss(cls_logits.float(), cls_target, patch_weights)
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
                'cls_loss': cls_loss.detach().cpu().numpy(),
                # tracks how much of each batch actually carries subtype evidence. If this sits near
                # the floor, the sampler is rarely hitting the lesion and oversampling needs raising.
                'cls_patch_weight': float(patch_weights.mean())}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        super().on_train_epoch_end(train_outputs)
        self.logger.log('train_losses_seg', float(np.mean(outputs['seg_loss'])), self.current_epoch)
        self.logger.log('train_losses_cls', float(np.mean(outputs['cls_loss'])), self.current_epoch)
        self.logger.log_metrics_dict({'train/cls_loss_weight': self._current_cls_weight(),
                                      'train/cls_patch_weight': float(np.mean(outputs['cls_patch_weight']))},
                                     step=self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        keys = list(batch['keys'])
        cls_target = self._subtype_targets(keys)
        seg_target = batch['target'][0] if isinstance(batch['target'], list) else batch['target']
        patch_weights = self._patch_weights(seg_target.to(self.device, non_blocking=True))

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
        cls_loss = self._weighted_cls_loss(cls_logits, cls_target, patch_weights)
        # the parent's 'loss' is the segmentation loss only; report both parts and the combined value
        out['seg_loss'] = out['loss']
        out['cls_loss'] = cls_loss.detach().cpu().numpy()
        out['loss'] = out['loss'] + self._current_cls_weight() * out['cls_loss']
        out['cls_probs'] = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
        out['cls_target'] = cls_target.detach().cpu().numpy()
        # case identifiers so on_validation_epoch_end can aggregate patches back into cases
        out['cls_keys'] = keys
        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)

        self.logger.log('val_losses_seg', float(np.mean(outputs['seg_loss'])), self.current_epoch)
        self.logger.log('val_losses_cls', float(np.mean(outputs['cls_loss'])), self.current_epoch)

        probs = np.concatenate([np.atleast_2d(p) for p in outputs['cls_probs']], axis=0)
        targets = np.concatenate([np.atleast_1d(t) for t in outputs['cls_target']], axis=0)

        patch_metrics = classification_metrics(targets, probs, num_classes=self.num_subtypes,
                                               prefix='val_patch_cls/')

        case_targets, case_probs = self._aggregate_by_case(outputs['cls_keys'], probs, targets)
        case_metrics = classification_metrics(case_targets, case_probs, num_classes=self.num_subtypes,
                                              prefix='val_case_cls/')

        # headline metrics also go to the local log so they appear in progress.png and survive resume.
        # The cls_* keys hold the CASE-level values: that is the level the quiz is scored at and the
        # level convergence detection should watch. Patch-level macro-F1 is kept alongside as a
        # diagnostic - it is bounded below the case-level number by construction, because it counts
        # patches whose subtype label is not inferable from the patch.
        self.logger.log('cls_balanced_accuracy', case_metrics['val_case_cls/balanced_accuracy'], self.current_epoch)
        self.logger.log('cls_macro_f1', case_metrics['val_case_cls/macro_f1'], self.current_epoch)
        self.logger.log('cls_mcc', case_metrics['val_case_cls/mcc'], self.current_epoch)
        self.logger.log('cls_macro_f1_patch', patch_metrics['val_patch_cls/macro_f1'], self.current_epoch)
        self.logger.log_metrics_dict({**patch_metrics, **case_metrics,
                                      'val_case_cls/n_cases': float(len(case_targets))},
                                     step=self.current_epoch)

    @staticmethod
    def _aggregate_by_case(keys: List[str], probs: np.ndarray, targets: np.ndarray):
        """
        Average the patch probabilities belonging to each case into one prediction per case.

        This is the training-time mirror of the aggregation in predict_quiz.py, and it reports at the
        level the quiz is actually scored at: a case gets one subtype, not one per patch.

        Uniform averaging is used here - predict_quiz's --uniform_pooling mode - rather than the
        foreground-weighted variant it uses by default. Weighting there is driven by the *predicted*
        segmentation, which is not available per patch at this point without a second pass, and using
        the ground-truth weights instead would make the monitored number optimistic relative to what
        inference can achieve. Uniform pooling is the conservative choice: the weighted variant is what
        actually gets submitted and scores at least as well.

        Note this covers only the cases the sampler happened to draw this epoch, so the case count
        varies from epoch to epoch. That is why the convergence detector smooths its input.
        """
        sums, counts, case_target = {}, {}, {}
        for key, prob, target in zip(keys, probs, targets):
            sums[key] = sums[key] + prob if key in sums else prob.astype(np.float64)
            counts[key] = counts.get(key, 0) + 1
            case_target[key] = target
        ordered = sorted(sums)
        if not ordered:
            return np.zeros(0, dtype=np.int64), np.zeros((0, probs.shape[-1]), dtype=np.float64)
        case_probs = np.stack([sums[k] / counts[k] for k in ordered], axis=0)
        case_targets = np.array([case_target[k] for k in ordered], dtype=np.int64)
        return case_targets, case_probs

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

    # ------------------------------------------------------------------ convergence

    def _monitored_value(self) -> float:
        """
        The scalar the convergence detector watches. Higher is better, roughly in [0, 1].

        'cls_macro_f1' is the case-level value (see on_validation_epoch_end). Watching the patch-level
        one instead would track a metric the model is not optimised for and cannot reach, since a large
        share of patches carry no evidence for their own label.
        """
        segmentation = float(self.logger.get_value('ema_fg_dice', -1))
        classification = float(self.logger.get_value('cls_macro_f1', -1))
        if self.convergence_metric == 'ema_fg_dice':
            return segmentation
        if self.convergence_metric == 'cls_macro_f1':
            return classification
        if self.convergence_metric == 'combined':
            # both are in [0, 1] so a plain mean is meaningful. Using segmentation alone would stop
            # while classification is still improving: it has a warmup and only 252 case labels, so
            # it converges later than segmentation.
            return 0.5 * (segmentation + classification)
        raise ValueError(f'unknown convergence_metric {self.convergence_metric!r}')

    def _start_anneal_out(self, anneal_start_epoch: Optional[int] = None, start_lr: Optional[float] = None):
        """
        Replace the poly schedule with a decay from the current LR to ~0 over anneal_epochs.

        Epoch bookkeeping: this runs from on_epoch_end, which is called *after*
        nnUNetTrainer.on_epoch_end has already incremented current_epoch. So self.current_epoch is the
        index of the next epoch that will run, and the anneal covers epochs
        [anneal_start, anneal_start + anneal_epochs - 1] inclusive. The polynomial denominator is
        anneal_epochs - 1 so that the *last epoch actually executed* sees lr = 0, not the epoch after
        the run has already stopped.

        The arguments are supplied when restoring a resumed run, so that resuming mid-anneal continues
        the original schedule instead of restarting (and thereby extending) it.
        """
        if self.convergence_anneal_epochs < 2:
            raise ValueError('convergence_anneal_epochs must be >= 2 for the learning rate to actually '
                             f'decay; got {self.convergence_anneal_epochs}')

        self._anneal_start_epoch = anneal_start_epoch if anneal_start_epoch is not None else self.current_epoch
        self._final_epoch = self._anneal_start_epoch + self.convergence_anneal_epochs
        current_lr = start_lr if start_lr is not None else self.optimizer.param_groups[0]['lr']
        self._anneal_start_lr = current_lr
        self.lr_scheduler = AnnealOutLRScheduler(
            self.optimizer,
            start_lr=current_lr,
            start_epoch=self._anneal_start_epoch,
            num_epochs=self.convergence_anneal_epochs - 1,
        )
        self.print_to_log_file(
            f'Convergence detected at epoch {self.current_epoch}. Best smoothed '
            f'{self.convergence_metric} = {np.round(self.convergence_detector.best_value, 4)} at epoch '
            f'{self.convergence_detector.best_epoch}. Annealing the learning rate from '
            f'{np.round(current_lr, 6)} to 0 over epochs '
            f'{self._anneal_start_epoch}-{self._final_epoch - 1}, then stopping.')

    def on_epoch_end(self):
        super().on_epoch_end()
        self.print_to_log_file(
            'cls (case-level, internal val): '
            f"bal_acc {np.round(self.logger.get_value('cls_balanced_accuracy', -1), 4)}, "
            f"macro_F1 {np.round(self.logger.get_value('cls_macro_f1', -1), 4)}, "
            f"MCC {np.round(self.logger.get_value('cls_mcc', -1), 4)} "
            f"(patch-level macro_F1 {np.round(self.logger.get_value('cls_macro_f1_patch', -1), 4)})")

        newly_converged = self.convergence_detector.update(self.current_epoch, self._monitored_value())
        detector = self.convergence_detector
        self.logger.log_metrics_dict({
            'convergence/monitored': self._monitored_value(),
            'convergence/smoothed': detector.smoothed_value,
            'convergence/best': detector.best_value,
            'convergence/epochs_without_improvement': detector.epochs_without_improvement,
        }, step=self.current_epoch)

        if newly_converged:
            self.logger.log_summary('convergence/detected_at_epoch', self.current_epoch)
            self.logger.log_summary('convergence/best_epoch', detector.best_epoch)
            if self.enable_early_stopping:
                self._start_anneal_out()
            else:
                self.print_to_log_file(
                    f'Convergence detected at epoch {self.current_epoch} (best epoch '
                    f'{detector.best_epoch}). early stopping is disabled, so training continues to '
                    f'epoch {self.num_epochs}. Consider setting num_epochs near '
                    f'{detector.best_epoch + self.convergence_anneal_epochs} for the next run so the '
                    f'learning rate anneals fully within the budget.')

        self._save_convergence_state()

        if self._final_epoch is not None and self.current_epoch >= self._final_epoch:
            self.print_to_log_file(f'Anneal-out complete at epoch {self.current_epoch}. Stopping.')
            # current_epoch is normally incremented by the parent's on_epoch_end; unwinding here skips
            # the rest of run_training's loop. See ConvergenceReached for why this uses an exception.
            raise ConvergenceReached()

    def run_training(self):
        try:
            super().run_training()
        except ConvergenceReached:
            # the parent's run_training never reached its own on_train_end because we unwound its loop
            self.on_train_end()

    # ------------------------------------------------------------------ convergence persistence

    @property
    def convergence_state_file(self) -> str:
        return join(self.output_folder, 'convergence_state.json')

    def _save_convergence_state(self):
        """
        Persisted next to the checkpoints rather than inside them: the state is a handful of scalars,
        and round-tripping a multi-hundred-MB checkpoint just to add them would be wasteful.
        """
        if self.local_rank != 0 or self.disable_checkpointing or self.output_folder is None:
            return
        save_json({'detector': self.convergence_detector.state_dict(),
                   'anneal_start_epoch': self._anneal_start_epoch,
                   'anneal_start_lr': self._anneal_start_lr,
                   'final_epoch': self._final_epoch}, self.convergence_state_file, sort_keys=False)

    def _restore_convergence_state(self):
        if self.output_folder is None or not isfile(self.convergence_state_file):
            return
        state = load_json(self.convergence_state_file)
        self.convergence_detector.load_state_dict(state['detector'])
        final_epoch = state.get('final_epoch')
        if final_epoch is not None and self.current_epoch < final_epoch:
            # resuming mid-anneal: rebuild the *original* schedule (same start epoch and start LR) so
            # the anneal continues rather than restarting from the resumed epoch and running longer
            self._start_anneal_out(anneal_start_epoch=state['anneal_start_epoch'],
                                   start_lr=state['anneal_start_lr'])
            self.print_to_log_file(f'Resumed mid-anneal; will still stop at epoch {self._final_epoch}.')

    def on_train_start(self):
        super().on_train_start()
        if self.current_epoch > 0:
            self._restore_convergence_state()
