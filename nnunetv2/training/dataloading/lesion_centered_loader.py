"""nnU-Net patch loader that centres every forced-foreground crop on the lesion label."""

from typing import Tuple, Union

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class LesionCenteredDataLoader(nnUNetDataLoader):
    """Select class-2 coordinates instead of choosing randomly among foreground classes."""

    def __init__(self, *args, lesion_label: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lesion_label = int(lesion_label)

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        requested_class = overwrite_class
        if force_fg:
            if class_locations is None or self.lesion_label not in class_locations:
                raise RuntimeError(
                    f'lesion-centred sampling requires class_locations for label {self.lesion_label}')
            if len(class_locations[self.lesion_label]) == 0:
                raise RuntimeError(
                    f'lesion-centred sampling found no voxels for label {self.lesion_label}')
            requested_class = self.lesion_label
        return super().get_bbox(
            data_shape,
            force_fg,
            class_locations,
            overwrite_class=requested_class,
            verbose=verbose,
        )
