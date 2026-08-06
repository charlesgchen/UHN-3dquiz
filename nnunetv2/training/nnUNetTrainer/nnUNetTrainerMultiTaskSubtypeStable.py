"""Numerically stable multi-task trainer for the pancreas subtype quiz.

The original multi-task trainer uses nnU-Net's SGD learning rate (``1e-2``)
for both the mature 102M-parameter segmentation network and the randomly
initialized cross-attention classification head.  In the first live run the
classification CE grew from roughly 1 to more than 50 during its warmup on
all three folds, while macro-F1 stayed at chance.  Because the classification
loss shares the optimizer with the segmentation task, that divergence also
starts to dominate the encoder gradients as the warmup weight increases.

Scaling the auxiliary loss to 0.05 gives the classification path an effective
SGD step size of 5e-4 at full warmup while preserving joint gradients through
the shared encoder.  The architecture, data, segmentation loss, augmentation,
class weighting, patch weighting, and convergence policy remain unchanged.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTaskSubtype import (
    nnUNetTrainerMultiTaskSubtype,
)


class nnUNetTrainerMultiTaskSubtypeStable(nnUNetTrainerMultiTaskSubtype):
    """Multi-task trainer with a stable auxiliary classification gradient scale."""

    cls_loss_weight: float = 0.05
