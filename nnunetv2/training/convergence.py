"""
Convergence detection and graceful early stopping for nnU-Net training.

Why this is not just "stop when the metric plateaus"
----------------------------------------------------
nnU-Net anneals the learning rate with a polynomial schedule over the *full* num_epochs:

    lr = initial_lr * (1 - epoch / num_epochs) ** 0.9

Killing a 1000-epoch run at epoch 300 leaves the learning rate at 72% of initial. The weights are
still bouncing around a wide basin and have never been annealed into it, so the model you keep is
materially worse than one that ran the schedule out - early stopping this way costs accuracy rather
than just saving time.

So convergence detection here has two stages:

  1. DETECT   a plateau in the monitored metric (patience on a smoothed best-so-far).
  2. ANNEAL   instead of stopping immediately, decay the learning rate from wherever it currently is
              down to ~0 over `anneal_epochs`, then stop. The model gets a proper anneal into the
              basin it found, at a fraction of the remaining budget.

Detection also works in report-only mode (enable_early_stopping=False): the trainer logs the epoch at
which the metric stopped improving so you can pick a sensible num_epochs for the next run, which is
the more reliable way to use a fixed budget.
"""

from typing import Dict, Optional

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class ConvergenceDetector:
    """
    Plateau detector on a higher-is-better metric.

    The monitored value is exponentially smoothed before comparison because the per-epoch signals
    available here are noisy: pseudo-Dice comes from 50 random patches, and patch-level macro-F1 from
    a few hundred patches. Without smoothing, patience triggers on noise.

    Args:
        patience: epochs without a new best (on the smoothed value) before declaring convergence
        min_delta: how much the smoothed value must improve to count as a new best
        min_epochs: never declare convergence before this epoch, regardless of patience
        smoothing: EMA factor for the monitored value; 0 disables smoothing
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-3, min_epochs: int = 100,
                 smoothing: float = 0.9):
        if not 0 <= smoothing < 1:
            raise ValueError(f'smoothing must be in [0, 1), got {smoothing}')
        if patience < 1:
            raise ValueError(f'patience must be >= 1, got {patience}')
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.smoothing = smoothing

        self.smoothed_value: Optional[float] = None
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_without_improvement: int = 0
        self.converged_at_epoch: Optional[int] = None

    def update(self, epoch: int, value: float) -> bool:
        """
        Feed one epoch's metric. Returns True the first time convergence is declared.

        Non-finite values (a metric that is undefined this epoch) are ignored rather than treated as
        a failure to improve, so a transient nan cannot burn through the patience budget.
        """
        if value is None or not np.isfinite(value):
            return False

        if self.smoothed_value is None or self.smoothing == 0:
            self.smoothed_value = float(value)
        else:
            self.smoothed_value = self.smoothing * self.smoothed_value + (1 - self.smoothing) * float(value)

        if self.best_value is None or self.smoothed_value > self.best_value + self.min_delta:
            self.best_value = self.smoothed_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if (self.converged_at_epoch is None
                and epoch + 1 >= self.min_epochs
                and self.epochs_without_improvement >= self.patience):
            self.converged_at_epoch = epoch
            return True
        return False

    @property
    def has_converged(self) -> bool:
        return self.converged_at_epoch is not None

    def state_dict(self) -> Dict:
        return {
            'smoothed_value': self.smoothed_value,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'converged_at_epoch': self.converged_at_epoch,
        }

    def load_state_dict(self, state: Dict):
        for key, value in state.items():
            setattr(self, key, value)


class AnnealOutLRScheduler(_LRScheduler):
    """
    Polynomial decay from `start_lr` to 0 over `num_epochs`, starting at absolute epoch `start_epoch`.

    Drop-in for PolyLRScheduler: nnUNetTrainer calls step(current_epoch) with the absolute epoch
    number, so this scheduler subtracts start_epoch itself.
    """

    def __init__(self, optimizer, start_lr: float, start_epoch: int, num_epochs: int,
                 exponent: float = 0.9):
        if num_epochs < 1:
            raise ValueError(f'num_epochs must be >= 1, got {num_epochs}')
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        progress = np.clip((current_step - self.start_epoch) / self.num_epochs, 0.0, 1.0)
        new_lr = self.start_lr * (1 - progress) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class ConvergenceReached(Exception):
    """
    Raised from on_epoch_end to unwind nnUNetTrainer.run_training's loop.

    The loop bound in run_training is `range(self.current_epoch, self.num_epochs)`, evaluated once, so
    mutating num_epochs mid-run cannot end it. Raising lets us reuse the upstream loop verbatim
    instead of copying it into a subclass where it would drift from upstream.
    """
