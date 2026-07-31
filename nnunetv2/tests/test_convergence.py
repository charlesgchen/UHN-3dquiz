import numpy as np
import pytest
import torch

from nnunetv2.training.convergence import (
    AnnealOutLRScheduler,
    ConvergenceDetector,
    ConvergenceReached,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


# ----------------------------------------------------------------- detector

def test_no_convergence_while_improving():
    detector = ConvergenceDetector(patience=5, min_delta=1e-4, min_epochs=0, smoothing=0.0)
    for epoch in range(50):
        assert not detector.update(epoch, 0.5 + 0.01 * epoch)
    assert not detector.has_converged


def test_converges_on_a_flat_plateau():
    detector = ConvergenceDetector(patience=5, min_delta=1e-4, min_epochs=0, smoothing=0.0)
    converged_at = None
    for epoch in range(30):
        if detector.update(epoch, 0.8):
            converged_at = epoch
            break
    assert converged_at is not None
    # first update sets the best, the next `patience` updates fail to improve
    assert converged_at == 5
    assert detector.has_converged


def test_min_epochs_blocks_early_convergence():
    detector = ConvergenceDetector(patience=2, min_delta=1e-4, min_epochs=20, smoothing=0.0)
    fired = [epoch for epoch in range(40) if detector.update(epoch, 0.8)]
    assert fired == [19], 'convergence must not be declared before min_epochs'


def test_min_delta_ignores_negligible_improvement():
    """Improvements smaller than min_delta must not reset the patience counter."""
    detector = ConvergenceDetector(patience=5, min_delta=0.01, min_epochs=0, smoothing=0.0)
    converged = False
    for epoch in range(20):
        converged = detector.update(epoch, 0.8 + 0.0001 * epoch)
        if converged:
            break
    assert converged, 'tiny improvements should still count as a plateau'


def test_smoothing_suppresses_noise():
    """A noisy but flat signal should still be detected as converged."""
    rng = np.random.default_rng(0)
    detector = ConvergenceDetector(patience=15, min_delta=1e-3, min_epochs=0, smoothing=0.9)
    converged = False
    for epoch in range(200):
        converged = detector.update(epoch, 0.8 + rng.normal(0, 0.05))
        if converged:
            break
    assert converged


def test_non_finite_values_do_not_burn_patience():
    """A transient nan (e.g. a metric undefined that epoch) must not count as 'no improvement'."""
    detector = ConvergenceDetector(patience=3, min_delta=1e-4, min_epochs=0, smoothing=0.0)
    detector.update(0, 0.8)
    for epoch in range(1, 20):
        assert not detector.update(epoch, float('nan'))
    assert detector.epochs_without_improvement == 0
    assert not detector.has_converged


def test_converges_only_once():
    detector = ConvergenceDetector(patience=2, min_delta=1e-4, min_epochs=0, smoothing=0.0)
    fired = [epoch for epoch in range(30) if detector.update(epoch, 0.8)]
    assert len(fired) == 1, 'update() should report convergence exactly once'


def test_state_dict_roundtrip():
    detector = ConvergenceDetector(patience=5, min_delta=1e-4, min_epochs=0, smoothing=0.5)
    for epoch in range(10):
        detector.update(epoch, 0.5 + 0.01 * epoch)

    restored = ConvergenceDetector(patience=5, min_delta=1e-4, min_epochs=0, smoothing=0.5)
    restored.load_state_dict(detector.state_dict())
    assert restored.best_value == detector.best_value
    assert restored.best_epoch == detector.best_epoch
    assert restored.smoothed_value == detector.smoothed_value
    assert restored.epochs_without_improvement == detector.epochs_without_improvement


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ConvergenceDetector(smoothing=1.0)
    with pytest.raises(ValueError):
        ConvergenceDetector(patience=0)


# ----------------------------------------------------------------- anneal-out scheduler

def _lr_after(scheduler, optimizer, epoch):
    scheduler.step(epoch)
    return optimizer.param_groups[0]['lr']


def test_anneal_out_decays_to_zero():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = AnnealOutLRScheduler(optimizer, start_lr=0.007, start_epoch=300, num_epochs=25)

    assert np.isclose(_lr_after(scheduler, optimizer, 300), 0.007)
    mid = _lr_after(scheduler, optimizer, 312)
    assert 0 < mid < 0.007
    assert np.isclose(_lr_after(scheduler, optimizer, 325), 0.0, atol=1e-9)


def test_anneal_out_clamps_outside_its_window():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = AnnealOutLRScheduler(optimizer, start_lr=0.007, start_epoch=300, num_epochs=25)

    assert np.isclose(_lr_after(scheduler, optimizer, 250), 0.007), 'before the window: hold start_lr'
    assert np.isclose(_lr_after(scheduler, optimizer, 400), 0.0, atol=1e-9), 'after: stay at 0'


def test_anneal_out_beats_truncated_poly_schedule():
    """
    The motivating comparison: stopping a 1000-epoch poly schedule at epoch 300 leaves the LR at ~72%
    of initial, whereas the anneal-out phase actually lands the model at ~0.
    """
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    poly = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=1000)
    lr_at_truncation = _lr_after(poly, optimizer, 300)
    assert lr_at_truncation > 0.007, 'poly schedule has barely annealed by epoch 300'

    anneal = AnnealOutLRScheduler(optimizer, start_lr=lr_at_truncation, start_epoch=300, num_epochs=25)
    assert np.isclose(_lr_after(anneal, optimizer, 325), 0.0, atol=1e-9)


def test_anneal_out_rejects_zero_length():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    with pytest.raises(ValueError):
        AnnealOutLRScheduler(optimizer, start_lr=0.01, start_epoch=0, num_epochs=0)


def test_convergence_reached_is_an_exception():
    assert issubclass(ConvergenceReached, Exception)
