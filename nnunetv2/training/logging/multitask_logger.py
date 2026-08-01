"""
Logging for the multi-task (segmentation + subtype classification) quiz trainer.

This extends the stock nnU-Net logging rather than modifying it, so vanilla trainings are unaffected:

  * MultiTaskLocalLogger adds per-task loss curves and the headline classification metrics to the
    epoch-indexed local log (these are checkpointed, so they survive a resume) and draws them in
    progress.png.
  * MultiTaskMetaLogger swaps that local logger in and adds log_metrics_dict(), which pushes the full
    Metrics Reloaded suite (per-class sensitivity/specificity/F-beta, AUROC, average precision, ...)
    to W&B only. Those are not put in the local log because LocalLogger is a fixed-key,
    exactly-one-value-per-epoch structure and stuffing dozens of optional keys into it would break
    both progress.png and checkpoint compatibility.

W&B is controlled by the environment variables the fork already defines:
    nnUNet_wandb_enabled = 1
    nnUNet_wandb_project = <project name>
    nnUNet_wandb_mode    = online | offline
"""

from typing import Any, Dict

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from nnunetv2.training.logging.nnunet_logger import LocalLogger, MetaLogger

# keys the stock LocalLogger already tracks; used to work out the current epoch safely
CORE_KEYS = ('mean_fg_dice', 'ema_fg_dice', 'dice_per_class_or_region', 'train_losses',
             'val_losses', 'lrs', 'epoch_start_timestamps', 'epoch_end_timestamps')

# additional epoch-indexed keys owned by this logger. The cls_* metrics are case-level (patches
# aggregated back into cases, which is how the quiz is scored); cls_macro_f1_patch is the patch-level
# diagnostic kept alongside it.
MULTITASK_KEYS = ('train_losses_seg', 'train_losses_cls', 'val_losses_seg', 'val_losses_cls',
                  'cls_balanced_accuracy', 'cls_macro_f1', 'cls_mcc', 'cls_macro_f1_patch')


class MultiTaskLocalLogger(LocalLogger):
    """LocalLogger plus per-task losses and headline classification metrics."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        for key in MULTITASK_KEYS:
            self.my_fantastic_logging[key] = list()

    def load_checkpoint(self, checkpoint: dict):
        """Restore, tolerating checkpoints written before the multi-task keys existed."""
        super().load_checkpoint(checkpoint)
        for key in MULTITASK_KEYS:
            if key not in self.my_fantastic_logging:
                self.my_fantastic_logging[key] = list()

    def _current_epoch(self) -> int:
        """Epoch inferred from the core keys only, so a half-populated new key cannot break plotting."""
        return min(len(self.my_fantastic_logging[k]) for k in CORE_KEYS) - 1

    def plot_progress_png(self, output_folder):
        epoch = self._current_epoch()
        if epoch < 0:
            return
        x_values = list(range(epoch + 1))

        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(5, 1, figsize=(30, 90))

        # panel 0: total loss + pseudo dice (same as stock nnU-Net)
        ax = ax_all[0]
        ax2 = ax.twinx()
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-',
                label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-',
                label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted',
                 label="pseudo dice", linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-',
                 label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # panel 1: the two task losses separately, so you can see which task is still improving
        ax = ax_all[1]
        for key, color, label in (('train_losses_seg', 'b', 'seg loss (train)'),
                                  ('val_losses_seg', 'r', 'seg loss (val)'),
                                  ('train_losses_cls', 'c', 'cls loss (train)'),
                                  ('val_losses_cls', 'm', 'cls loss (val)')):
            values = self.my_fantastic_logging[key]
            if len(values) >= epoch + 1:
                ax.plot(x_values, values[:epoch + 1], color=color, ls='-', label=label, linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=(0, 1))

        # panel 2: classification metrics on the internal validation split. Solid = case-level (what
        # the quiz scores and what convergence detection watches), dotted = patch-level diagnostic.
        ax = ax_all[2]
        for key, color, label, style in (('cls_balanced_accuracy', 'b', 'balanced accuracy (case)', '-'),
                                         ('cls_macro_f1', 'g', 'macro F1 (case)', '-'),
                                         ('cls_mcc', 'r', 'MCC (case)', '-'),
                                         ('cls_macro_f1_patch', 'g', 'macro F1 (patch)', 'dotted')):
            values = self.my_fantastic_logging[key]
            if len(values) >= epoch + 1:
                ax.plot(x_values, values[:epoch + 1], color=color, ls=style, label=label, linewidth=3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("epoch")
        ax.set_ylabel("classification metric")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=(0, 1))

        # panel 3: epoch duration
        ax = ax_all[3]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1],
                color='b', ls='-', label="epoch duration", linewidth=4)
        ax.set(ylim=[0, ax.get_ylim()[1]])
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # panel 4: learning rate
        ax = ax_all[4]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-',
                label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()


class MultiTaskMetaLogger(MetaLogger):
    """MetaLogger that uses MultiTaskLocalLogger and can forward arbitrary metric dicts to W&B."""

    def __init__(self, output_folder, resume, verbose: bool = False):
        super().__init__(output_folder, resume, verbose)
        # replace the local logger created by the parent with the multi-task aware one
        self.local_logger = MultiTaskLocalLogger(verbose)

    @classmethod
    def adopt(cls, meta_logger: MetaLogger, verbose: bool = False) -> 'MultiTaskMetaLogger':
        """
        Upgrade an already-constructed MetaLogger in place of building a new one.

        nnUNetTrainer.__init__ constructs a MetaLogger before a subclass gets control. Constructing a
        second one would call wandb.init() a second time (two runs for one training) and, because
        WandbLogger deletes the run directory when resume is False, would destroy the first run's data.
        So we reuse the existing W&B logger objects and only swap the local logger.
        """
        adopted = cls.__new__(cls)
        adopted.output_folder = meta_logger.output_folder
        adopted.resume = meta_logger.resume
        adopted.loggers = meta_logger.loggers          # reuse, do not re-init
        adopted.local_logger = MultiTaskLocalLogger(verbose)
        return adopted

    def log_metrics_dict(self, metrics: Dict[str, Any], step: int):
        """
        Send a flat dict of metrics to every non-local logger (i.e. W&B).

        Used for the full Metrics Reloaded suite, which is too wide and too optional to live in the
        fixed-key local log. Values that are None are skipped; NaNs are passed through so that a gap
        in a W&B chart correctly reads as "undefined this epoch" rather than as a zero.
        """
        for key, value in metrics.items():
            if value is None:
                continue
            for logger in self.loggers:
                logger.log(key, value, step)
