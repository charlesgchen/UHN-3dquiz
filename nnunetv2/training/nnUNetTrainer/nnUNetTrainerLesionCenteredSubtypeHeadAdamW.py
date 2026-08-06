"""Cross-attention subtype head trained exclusively on lesion-centred ROI patches.

The reset-head five-fold ensemble is the strongest classifier so far, but its ordinary nnU-Net
sampler still assigns case labels to many patches where the lesion is absent.  This trainer retains
the exact same frozen encoder, cross-attention head, optimizer, regularization, and internal
classification checkpoint policy while forcing every train and validation crop to contain label 2.

At whole-volume inference the standard sliding-window predictor remains unchanged except that patch
votes are weighted by predicted lesion mass, matching the evidence distribution used for training.
"""

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.lesion_centered_loader import LesionCenteredDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSubtypeHeadAdamW import (
    nnUNetTrainerSubtypeHeadAdamW,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainerLesionCenteredSubtypeHeadAdamW(nnUNetTrainerSubtypeHeadAdamW):
    """Fit the original cross-attention head with one label-2-centred ROI per sample."""

    lesion_center_label: int = 2

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 1.0

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        network = nnUNetTrainerSubtypeHeadAdamW.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        # MultiTaskPredictor recognizes this hint and weights patch votes by predicted label-2 mass.
        network.classification_pooling_label = nnUNetTrainerLesionCenteredSubtypeHeadAdamW.lesion_center_label
        return network

    def initialize(self):
        super().initialize()
        self.logger.update_config({
            'sampling_variant': 'every_patch_centered_on_training_lesion_label',
            'lesion_center_label': self.lesion_center_label,
            'lesion_centered_fraction': self.oversample_foreground_percent,
            'classification_architecture': 'original_cross_attention_pooling',
            'inference_patch_evidence': 'predicted_lesion_mass',
        })

    def get_dataloaders(self):
        """Use native augmentation around lesion-centred train and validation loaders."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_da, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        train_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_da,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_train, dataset_validation = self.get_tr_and_val_datasets()
        loader_kwargs = {
            'label_manager': self.label_manager,
            'oversample_foreground_percent': self.oversample_foreground_percent,
            'sampling_probabilities': None,
            'pad_sides': None,
            'probabilistic_oversampling': self.probabilistic_oversampling,
            'lesion_label': self.lesion_center_label,
        }
        loader_train = LesionCenteredDataLoader(
            dataset_train,
            self.batch_size,
            initial_patch_size,
            patch_size,
            transforms=train_transforms,
            **loader_kwargs,
        )
        loader_validation = LesionCenteredDataLoader(
            dataset_validation,
            self.batch_size,
            patch_size,
            patch_size,
            transforms=validation_transforms,
            **loader_kwargs,
        )

        allowed_processes = get_allowed_n_proc_DA()
        if allowed_processes == 0:
            augmenter_train = SingleThreadedAugmenter(loader_train, None)
            augmenter_validation = SingleThreadedAugmenter(loader_validation, None)
        else:
            augmenter_train = NonDetMultiThreadedAugmenter(
                data_loader=loader_train,
                transform=None,
                num_processes=allowed_processes,
                num_cached=max(6, allowed_processes // 2),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002,
            )
            augmenter_validation = NonDetMultiThreadedAugmenter(
                data_loader=loader_validation,
                transform=None,
                num_processes=max(1, allowed_processes // 2),
                num_cached=max(3, allowed_processes // 4),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002,
            )
        _ = next(augmenter_train)
        _ = next(augmenter_validation)
        return augmenter_train, augmenter_validation
