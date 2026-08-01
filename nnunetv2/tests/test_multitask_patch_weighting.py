"""
Tests for the per-patch weighting of the classification loss and for case-level metric aggregation.

These deliberately do not need the preprocessed dataset (unlike test_multitask_trainer.py, which skips
without it). The logic under test is arithmetic on tensors and dicts, so the trainer is instantiated
via __new__ with only the attributes each method actually reads. That keeps the tests fast and, more
importantly, means they run on the dev machine where the smoke tests cannot.
"""

import numpy as np
import pytest
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTaskSubtype import nnUNetTrainerMultiTaskSubtype

NUM_SUBTYPES = 3
LESION_LABEL = 2


def _stub(class_weights=(0.8, 1.0, 1.2), label_smoothing=0.1, patch_weighting=True, floor=0.3):
    """A trainer with only the attributes the weighting helpers read."""
    trainer = nnUNetTrainerMultiTaskSubtype.__new__(nnUNetTrainerMultiTaskSubtype)
    trainer.num_subtypes = NUM_SUBTYPES
    trainer.lesion_label = LESION_LABEL
    trainer.cls_patch_weighting = patch_weighting
    trainer.cls_patch_weight_floor = floor
    trainer.cls_class_weights = torch.tensor(class_weights, dtype=torch.float32)
    trainer.cls_loss = nn.CrossEntropyLoss(weight=trainer.cls_class_weights,
                                           label_smoothing=label_smoothing, reduction='none')
    return trainer


def _seg(*patches):
    """Build a [B, 1, X, Y, Z] target from per-patch fill values / arrays."""
    return torch.stack([torch.as_tensor(p, dtype=torch.float32).expand(1, 2, 2, 2).clone()
                        if np.isscalar(p) else torch.as_tensor(p, dtype=torch.float32).reshape(1, 2, 2, 2)
                        for p in patches])


# ------------------------------------------------------------------ _patch_weights

def test_patch_weight_tiers():
    """Background -> 0, pancreas only -> floor, any lesion voxel -> 1."""
    trainer = _stub(floor=0.3)
    background = _seg(0)
    pancreas_only = _seg(1)
    # a single lesion voxel among pancreas is enough to make the patch fully informative
    mixed = [1] * 8
    mixed[3] = LESION_LABEL

    weights = trainer._patch_weights(torch.cat([background, pancreas_only, _seg(mixed)]))
    assert weights.tolist() == pytest.approx([0.0, 0.3, 1.0])


def test_patch_weight_ignores_negative_ignore_label():
    """nnU-Net writes its ignore label negative; it must not count as foreground."""
    trainer = _stub()
    weights = trainer._patch_weights(_seg(-1))
    assert weights.tolist() == [0.0]


def test_patch_weighting_can_be_disabled():
    """cls_patch_weighting=False recovers the unweighted baseline, including empty patches."""
    trainer = _stub(patch_weighting=False)
    weights = trainer._patch_weights(torch.cat([_seg(0), _seg(1), _seg(LESION_LABEL)]))
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_patch_weight_floor_of_one_keeps_only_the_empty_patch_rule():
    trainer = _stub(floor=1.0)
    weights = trainer._patch_weights(torch.cat([_seg(0), _seg(1), _seg(LESION_LABEL)]))
    assert weights.tolist() == pytest.approx([0.0, 1.0, 1.0])


# ------------------------------------------------------------------ _weighted_cls_loss

def test_uniform_weights_match_pytorch_mean_reduction():
    """
    The scale-preservation invariant.

    PyTorch's weighted 'mean' reduction divides by the sum of the target classes' weights, not by the
    batch size. If _weighted_cls_loss got that denominator wrong, enabling patch weighting would
    silently rescale the classification loss and every tuned cls_loss_weight would change meaning.
    """
    torch.manual_seed(0)
    trainer = _stub()
    logits = torch.randn(6, NUM_SUBTYPES)
    targets = torch.tensor([0, 1, 2, 2, 1, 0])

    reference = nn.CrossEntropyLoss(weight=trainer.cls_class_weights, label_smoothing=0.1,
                                    reduction='mean')(logits, targets)
    ours = trainer._weighted_cls_loss(logits, targets, torch.ones(6))
    assert ours.item() == pytest.approx(reference.item(), rel=1e-6)


def test_uniform_weights_match_regardless_of_magnitude():
    """Any constant patch weight is the same weighted mean; only relative weights matter."""
    torch.manual_seed(1)
    trainer = _stub()
    logits = torch.randn(4, NUM_SUBTYPES)
    targets = torch.tensor([1, 0, 2, 1])

    at_one = trainer._weighted_cls_loss(logits, targets, torch.ones(4))
    at_third = trainer._weighted_cls_loss(logits, targets, torch.full((4,), 0.3))
    assert at_third.item() == pytest.approx(at_one.item(), rel=1e-6)


def test_zero_weighted_patches_are_excluded_exactly():
    """A batch of [informative, uninformative] must equal the informative sample on its own."""
    torch.manual_seed(2)
    trainer = _stub()
    logits = torch.randn(2, NUM_SUBTYPES)
    targets = torch.tensor([2, 0])

    both = trainer._weighted_cls_loss(logits, targets, torch.tensor([1.0, 0.0]))
    only_first = trainer._weighted_cls_loss(logits[:1], targets[:1], torch.ones(1))
    assert both.item() == pytest.approx(only_first.item(), rel=1e-6)


def test_all_zero_weights_give_zero_loss_and_finite_gradient():
    """
    A batch where no patch contains foreground must not divide by ~0.

    The loss has to stay attached to the graph so AMP's grad scaler and DDP still see a real tensor.
    """
    trainer = _stub()
    logits = torch.randn(2, NUM_SUBTYPES, requires_grad=True)
    targets = torch.tensor([0, 1])

    loss = trainer._weighted_cls_loss(logits, targets, torch.zeros(2))
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad == 0).all()


def test_relative_weighting_shifts_the_loss_toward_the_heavier_patch():
    """Down-weighting (not dropping) a patch moves the loss between the two extremes."""
    torch.manual_seed(3)
    trainer = _stub()
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    targets = torch.tensor([0, 0])  # first patch is confidently right, second confidently wrong

    lesion_first = trainer._weighted_cls_loss(logits, targets, torch.tensor([1.0, 0.3]))
    lesion_second = trainer._weighted_cls_loss(logits, targets, torch.tensor([0.3, 1.0]))
    uniform = trainer._weighted_cls_loss(logits, targets, torch.ones(2))
    assert lesion_first.item() < uniform.item() < lesion_second.item()


# ------------------------------------------------------------------ _aggregate_by_case

def test_aggregate_by_case_averages_patches_of_the_same_case():
    keys = ['quiz_1_000', 'quiz_1_000', 'quiz_0_007']
    probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.2, 0.2, 0.6]])
    targets = np.array([1, 1, 2])

    case_targets, case_probs = nnUNetTrainerMultiTaskSubtype._aggregate_by_case(keys, probs, targets)

    assert case_targets.tolist() == [2, 1]                       # sorted by case id
    assert case_probs[0].tolist() == pytest.approx([0.2, 0.2, 0.6])
    assert case_probs[1].tolist() == pytest.approx([0.6, 0.3, 0.1])


def test_aggregate_by_case_does_not_mutate_its_input():
    """The running sum must not accumulate into the caller's array."""
    keys = ['a', 'a']
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    original = probs.copy()
    nnUNetTrainerMultiTaskSubtype._aggregate_by_case(keys, probs, np.array([0, 0]))
    assert np.array_equal(probs, original)


def test_aggregate_by_case_handles_no_patches():
    case_targets, case_probs = nnUNetTrainerMultiTaskSubtype._aggregate_by_case(
        [], np.zeros((0, NUM_SUBTYPES)), np.zeros(0))
    assert case_targets.shape == (0,)
    assert case_probs.shape == (0, NUM_SUBTYPES)


def test_case_level_macro_f1_exceeds_patch_level_when_patches_are_noisy():
    """
    The reason case-level drives convergence: averaging patches of a case recovers the label even when
    individual patches are wrong, so the patch-level number understates real performance.
    """
    from nnunetv2.evaluation.quiz_metrics import classification_metrics

    keys, probs, targets = [], [], []
    for case in range(9):
        label = case % NUM_SUBTYPES
        # two patches lean correct, one leans wrong -> every case aggregates to the right class,
        # but a third of the patches are individually misclassified
        for prob in ([0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.1, 0.6, 0.3]):
            keys.append(f'case_{case:03d}')
            probs.append(np.roll(prob, label).tolist())
            targets.append(label)

    probs, targets = np.array(probs), np.array(targets)
    patch_f1 = classification_metrics(targets, probs, num_classes=NUM_SUBTYPES, prefix='p/')['p/macro_f1']

    case_targets, case_probs = nnUNetTrainerMultiTaskSubtype._aggregate_by_case(keys, probs, targets)
    case_f1 = classification_metrics(case_targets, case_probs, num_classes=NUM_SUBTYPES,
                                     prefix='c/')['c/macro_f1']

    assert case_f1 == pytest.approx(1.0)
    assert patch_f1 < case_f1


# ------------------------------------------------------------------ _resolve_lesion_label

def test_resolve_lesion_label_prefers_the_named_label():
    trainer = nnUNetTrainerMultiTaskSubtype.__new__(nnUNetTrainerMultiTaskSubtype)
    trainer.dataset_json = {'labels': {'background': 0, 'pancreas': 1, 'lesion': 2}}
    assert trainer._resolve_lesion_label() == 2


def test_resolve_lesion_label_falls_back_to_highest_foreground():
    trainer = nnUNetTrainerMultiTaskSubtype.__new__(nnUNetTrainerMultiTaskSubtype)
    trainer.dataset_json = {'labels': {'background': 0, 'organ': 1, 'tumour': 3}}
    assert trainer._resolve_lesion_label() == 3


def test_resolve_lesion_label_raises_without_any_foreground():
    trainer = nnUNetTrainerMultiTaskSubtype.__new__(nnUNetTrainerMultiTaskSubtype)
    trainer.dataset_json = {'labels': {'background': 0}}
    with pytest.raises(RuntimeError, match='lesion label'):
        trainer._resolve_lesion_label()
