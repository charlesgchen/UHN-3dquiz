"""
Tests for the inference-side pieces: classification aggregation across patches, the submission CSV
format, held-out evaluation, and packaging.

These avoid loading a trained model - the aggregation logic is tested directly on the accumulator.
"""

import csv
import json
import os
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from nnunetv2.evaluation.evaluate_quiz import (
    check_quiz_targets,
    evaluate_quiz_validation,
    evaluate_segmentation,
    package_submission,
)
from nnunetv2.inference.predict_quiz import (
    PROBABILITIES_JSON,
    SUBTYPE_CSV,
    MultiTaskPredictor,
    gamma_intensity_view,
    write_subtype_outputs,
)


class _Accumulator(MultiTaskPredictor):
    """Bare accumulator: skips nnUNetPredictor.__init__ so no model or plans are needed."""

    def __init__(self, foreground_weighted_pooling=True, classification_patch_pooling='weighted_mean'):
        self.foreground_weighted_pooling = foreground_weighted_pooling
        self.classification_patch_pooling = classification_patch_pooling
        self.save_classification_embeddings = False
        self.save_classification_patch_embeddings = False
        self._reset_classification_accumulator()


def _seg_logits(foreground_fraction, shape=(1, 3, 4, 4, 4)):
    """Segmentation logits whose softmax puts `foreground_fraction` of the patch on classes 1/2."""
    seg = torch.zeros(shape)
    n_voxels = int(np.prod(shape[2:]))
    n_fg = int(round(foreground_fraction * n_voxels))
    flat = seg.reshape(shape[0], shape[1], -1)
    flat[:, 0, :] = 20.0                 # background dominates everywhere
    flat[:, 0, :n_fg] = -20.0            # except in the foreground voxels
    flat[:, 1, :n_fg] = 20.0
    return flat.reshape(shape)


# ----------------------------------------------------------------- aggregation

def test_single_patch_returns_its_own_probabilities():
    acc = _Accumulator()
    logits = torch.log(torch.tensor([[0.1, 0.7, 0.2]]))
    acc._accumulate_classification(logits, _seg_logits(0.5))
    assert np.allclose(acc.get_case_classification(), [0.1, 0.7, 0.2], atol=1e-5)


def test_uniform_pooling_averages_patches():
    acc = _Accumulator(foreground_weighted_pooling=False)
    acc._accumulate_classification(torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), _seg_logits(0.9))
    acc._accumulate_classification(torch.log(torch.tensor([[0.0, 1.0, 0.0]]) + 1e-12), _seg_logits(0.1))
    assert np.allclose(acc.get_case_classification(), [0.5, 0.5, 0.0], atol=1e-4)


def test_foreground_weighting_downweights_background_patches():
    """A patch that is almost all background must not get an equal vote."""
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc._accumulate_classification(torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), _seg_logits(0.9))
    acc._accumulate_classification(torch.log(torch.tensor([[0.0, 1.0, 0.0]]) + 1e-12), _seg_logits(0.1))

    probabilities = acc.get_case_classification()
    assert probabilities[0] > 0.8, 'the foreground-rich patch should dominate'
    assert np.isclose(probabilities.sum(), 1.0)


def test_top1_pooling_selects_strongest_roi_evidence_patch():
    acc = _Accumulator(foreground_weighted_pooling=True, classification_patch_pooling='top1')
    acc.network = SimpleNamespace(classification_pooling_label=2)

    lesion_rich = _seg_logits(0.9)
    lesion_rich[:, 1], lesion_rich[:, 2] = lesion_rich[:, 2].clone(), lesion_rich[:, 1].clone()
    lesion_poor = _seg_logits(0.1)
    lesion_poor[:, 1], lesion_poor[:, 2] = lesion_poor[:, 2].clone(), lesion_poor[:, 1].clone()
    acc._accumulate_classification(
        torch.log(torch.tensor([[0.9, 0.1, 0.0]]) + 1e-12), lesion_rich)
    acc._accumulate_classification(
        torch.log(torch.tensor([[0.0, 0.1, 0.9]]) + 1e-12), lesion_poor)

    assert np.allclose(acc.get_case_classification(), [0.9, 0.1, 0.0], atol=1e-4)


def test_frozen_embeddings_use_same_roi_evidence_pooling_as_probabilities():
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc.save_classification_embeddings = True
    acc.network = SimpleNamespace(
        classification_pooling_label=2,
        last_classification_embedding=torch.tensor([[1.0, 3.0]]),
    )
    lesion_rich = _seg_logits(0.9)
    lesion_rich[:, 1], lesion_rich[:, 2] = lesion_rich[:, 2].clone(), lesion_rich[:, 1].clone()
    acc._accumulate_classification(torch.zeros(1, 3), lesion_rich)

    assert np.allclose(acc.get_case_classification_embeddings(), [[1.0, 3.0]], atol=1e-5)


def test_case_embeddings_are_returned_in_fold_member_order():
    acc = _Accumulator(foreground_weighted_pooling=False)
    acc.save_classification_embeddings = True
    acc.network = SimpleNamespace(last_classification_embedding=torch.tensor([[2.0]]))
    acc._active_classification_member = 1
    acc._accumulate_classification(torch.zeros(1, 3), _seg_logits(0.5))

    acc.network.last_classification_embedding = torch.tensor([[1.0]])
    acc._active_classification_member = 0
    acc._accumulate_classification(torch.zeros(1, 3), _seg_logits(0.5))

    assert np.allclose(acc.get_case_classification_embeddings(), [[1.0], [2.0]])


def test_patch_embeddings_preserve_each_patch_and_its_roi_evidence():
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc.save_classification_patch_embeddings = True
    acc.network = SimpleNamespace(
        classification_pooling_label=2,
        last_classification_embedding=torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )
    segmentation = _seg_logits(0.5, shape=(2, 3, 4, 4, 4))
    segmentation[:, 1], segmentation[:, 2] = (
        segmentation[:, 2].clone(), segmentation[:, 1].clone())
    acc._accumulate_classification(torch.zeros(2, 3), segmentation)

    members = acc.get_case_classification_patch_embeddings()
    assert len(members) == 1
    embeddings, weights = members[0]
    assert np.allclose(embeddings, [[1.0, 3.0], [2.0, 4.0]])
    assert weights.shape == (2,)
    assert np.all(weights > 0)


def test_network_can_request_lesion_weighted_patch_pooling():
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc.network = SimpleNamespace(classification_pooling_label=2)

    lesion_rich = _seg_logits(0.9)
    lesion_rich[:, 1], lesion_rich[:, 2] = lesion_rich[:, 2].clone(), lesion_rich[:, 1].clone()
    lesion_poor = _seg_logits(0.1)
    lesion_poor[:, 1], lesion_poor[:, 2] = lesion_poor[:, 2].clone(), lesion_poor[:, 1].clone()
    acc._accumulate_classification(
        torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), lesion_rich)
    acc._accumulate_classification(
        torch.log(torch.tensor([[0.0, 1.0, 0.0]]) + 1e-12), lesion_poor)

    probabilities = acc.get_case_classification()
    assert probabilities[0] > 0.8, 'the lesion-rich patch should dominate for an ROI head'


def test_network_rejects_invalid_classification_pooling_label():
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc.network = SimpleNamespace(classification_pooling_label=3)
    with pytest.raises(ValueError, match='outside segmentation channels'):
        acc._accumulate_classification(torch.zeros(1, 3), _seg_logits(0.5))


def test_falls_back_to_uniform_when_no_patch_has_foreground():
    """All-background case: weighting would divide by ~0, so it must degrade gracefully."""
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc._accumulate_classification(torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), _seg_logits(0.0))
    acc._accumulate_classification(torch.log(torch.tensor([[0.0, 0.0, 1.0]]) + 1e-12), _seg_logits(0.0))

    probabilities = acc.get_case_classification()
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.allclose(probabilities, [0.5, 0.0, 0.5], atol=1e-3)


def test_batched_patches_accumulate():
    acc = _Accumulator(foreground_weighted_pooling=False)
    logits = torch.log(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) + 1e-12)
    acc._accumulate_classification(logits, _seg_logits(0.5, shape=(2, 3, 4, 4, 4)))
    assert acc._cls_patch_count == 2
    assert np.allclose(acc.get_case_classification(), [0.5, 0.0, 0.5], atol=1e-4)


def test_fivefold_pooling_weights_models_equally_after_pooling_patches():
    """Foreground mass may weight patches within a fold, but must not weight folds themselves."""
    acc = _Accumulator(foreground_weighted_pooling=True)
    acc._active_classification_member = 0
    acc._accumulate_classification(
        torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), _seg_logits(0.9))
    acc._active_classification_member = 1
    acc._accumulate_classification(
        torch.log(torch.tensor([[0.0, 1.0, 0.0]]) + 1e-12), _seg_logits(0.1))

    assert np.allclose(acc.get_case_classification(), [0.5, 0.5, 0.0], atol=1e-4)


def test_fold_ensemble_loop_marks_each_model_for_classification_pooling():
    class _Network:
        def load_state_dict(self, state):
            self.member = state['member']

    class _FoldPredictor(_Accumulator):
        def __init__(self):
            super().__init__(foreground_weighted_pooling=True)
            self.network = _Network()
            self.list_of_parameters = [{'member': 0}, {'member': 1}]
            self.verbose = False

        def predict_sliding_window_return_logits(self, data):
            member = self.network.member
            probabilities = ([1.0, 0.0, 0.0] if member == 0 else [0.0, 1.0, 0.0])
            foreground = 0.9 if member == 0 else 0.1
            self._accumulate_classification(
                torch.log(torch.tensor([probabilities]) + 1e-12), _seg_logits(foreground))
            return torch.full((3, 2, 2, 2), float(member))

    predictor = _FoldPredictor()
    segmentation_logits = predictor.predict_logits_from_preprocessed_data(torch.empty(0))

    assert torch.allclose(segmentation_logits, torch.full_like(segmentation_logits, 0.5))
    assert np.allclose(predictor.get_case_classification(), [0.5, 0.5, 0.0], atol=1e-4)
    assert predictor._active_classification_member is None


def test_gamma_intensity_view_retains_per_channel_statistics():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((2, 2, 5, 6, 7), generator=generator)
    transformed = gamma_intensity_view(x, 0.85)

    assert not torch.allclose(transformed, x)
    assert torch.allclose(transformed.mean((2, 3, 4)), x.mean((2, 3, 4)), atol=1e-5)
    assert torch.allclose(
        transformed.std((2, 3, 4), correction=0),
        x.std((2, 3, 4), correction=0),
        atol=1e-5,
    )


def test_gamma_one_and_constant_inputs_are_unchanged():
    x = torch.randn((1, 1, 4, 4, 4))
    assert torch.allclose(gamma_intensity_view(x, 1.0), x, atol=1e-5)
    constant = torch.full((1, 1, 4, 4, 4), 3.0)
    assert torch.equal(gamma_intensity_view(constant, 0.85), constant)


def test_gamma_tta_adds_classification_votes_without_changing_segmentation():
    class _Network:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            segmentation = torch.zeros((len(x), 3, *x.shape[2:]))
            logits = torch.tensor([[0.0, float(self.calls), 0.0]]).repeat(len(x), 1)
            return segmentation, logits

    predictor = _Accumulator(foreground_weighted_pooling=False)
    predictor.network = _Network()
    predictor.use_mirroring = False
    predictor.allowed_mirroring_axes = None
    predictor.classification_gamma_tta = (0.85, 1.15)
    prediction = predictor._internal_maybe_mirror_and_predict(
        torch.randn((1, 1, 4, 4, 4))
    )

    assert predictor.network.calls == 3
    assert predictor._cls_patch_count == 3
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_accumulator_raises_before_any_patch():
    with pytest.raises(RuntimeError, match='no classification logits'):
        _Accumulator().get_case_classification()


def test_reset_clears_previous_case():
    acc = _Accumulator()
    acc._accumulate_classification(torch.log(torch.tensor([[1.0, 0.0, 0.0]]) + 1e-12), _seg_logits(0.5))
    acc._reset_classification_accumulator()
    with pytest.raises(RuntimeError):
        acc.get_case_classification()


# ----------------------------------------------------------------- submission format

def test_subtype_csv_matches_required_format(tmp_path):
    probabilities = {'quiz_037': np.array([0.8, 0.1, 0.1]),
                     'quiz_045': np.array([0.1, 0.7, 0.2]),
                     'quiz_047': np.array([0.2, 0.2, 0.6])}
    write_subtype_outputs(probabilities, str(tmp_path))

    with open(tmp_path / SUBTYPE_CSV, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['Names', 'Subtype']
    assert rows[1] == ['quiz_037.nii.gz', '0']
    assert rows[2] == ['quiz_045.nii.gz', '1']
    assert rows[3] == ['quiz_047.nii.gz', '2']


def test_probabilities_json_roundtrips(tmp_path):
    probabilities = {'quiz_037': np.array([0.8, 0.1, 0.1])}
    write_subtype_outputs(probabilities, str(tmp_path))
    with open(tmp_path / PROBABILITIES_JSON) as f:
        loaded = json.load(f)
    assert np.allclose(loaded['quiz_037'], [0.8, 0.1, 0.1])


# ----------------------------------------------------------------- evaluation

def _write_label_map(path, array, spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path), useCompression=True)


def test_evaluate_segmentation_perfect_predictions(tmp_path):
    ref_dir = tmp_path / 'ref'
    pred_dir = tmp_path / 'pred'
    ref_dir.mkdir()
    pred_dir.mkdir()

    for i in range(3):
        label = np.zeros((8, 8, 8), dtype=np.uint8)
        label[2:6, 2:6, 2:6] = 1
        label[3:5, 3:5, 3:5] = 2
        _write_label_map(ref_dir / f'quiz_0_{i:03d}.nii.gz', label)
        _write_label_map(pred_dir / f'quiz_0_{i:03d}.nii.gz', label)

    metrics = evaluate_segmentation(str(pred_dir), str(ref_dir))
    assert np.isclose(metrics['seg/dice_whole_pancreas_mean'], 1.0)
    assert np.isclose(metrics['seg/dice_lesion_mean'], 1.0)
    assert metrics['seg/dice_lesion_n'] == 3


def test_evaluate_segmentation_refuses_partial_predictions(tmp_path):
    """Silently scoring only the cases that happen to exist would inflate the reported number."""
    ref_dir = tmp_path / 'ref'
    pred_dir = tmp_path / 'pred'
    ref_dir.mkdir()
    pred_dir.mkdir()

    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 1
    for i in range(3):
        _write_label_map(ref_dir / f'quiz_0_{i:03d}.nii.gz', label)
    _write_label_map(pred_dir / 'quiz_0_000.nii.gz', label)

    with pytest.raises(RuntimeError, match='no prediction'):
        evaluate_segmentation(str(pred_dir), str(ref_dir))


def test_end_to_end_evaluation(tmp_path):
    raw = tmp_path / 'Dataset001_PancreasQuiz'
    (raw / 'labelsVal').mkdir(parents=True)
    pred = tmp_path / 'pred'
    pred.mkdir()

    subtypes = {}
    probabilities = {}
    for i, subtype in enumerate([0, 1, 2, 0, 1, 2]):
        case = f'quiz_{subtype}_{i:03d}'
        label = np.zeros((8, 8, 8), dtype=np.uint8)
        label[2:6, 2:6, 2:6] = 1
        label[3:5, 3:5, 3:5] = 2
        _write_label_map(raw / 'labelsVal' / f'{case}.nii.gz', label)
        _write_label_map(pred / f'{case}.nii.gz', label)
        subtypes[case] = subtype
        onehot = np.zeros(3)
        onehot[subtype] = 1.0
        probabilities[case] = onehot

    with open(raw / 'subtype_labels.json', 'w') as f:
        json.dump({'train': {}, 'validation': subtypes}, f)
    write_subtype_outputs(probabilities, str(pred))

    metrics = evaluate_quiz_validation(str(pred), str(raw))
    assert np.isclose(metrics['seg/dice_whole_pancreas_mean'], 1.0)
    assert np.isclose(metrics['cls/macro_f1'], 1.0)
    assert np.isclose(metrics['cls/mcc'], 1.0)
    assert all(check_quiz_targets(metrics, 'undergraduate').values())


def test_check_quiz_targets_thresholds():
    metrics = {'seg/dice_whole_pancreas_mean': 0.905, 'seg/dice_lesion_mean': 0.29, 'cls/macro_f1': 0.65}
    undergraduate = check_quiz_targets(metrics, 'undergraduate')
    assert all(undergraduate.values())
    master = check_quiz_targets(metrics, 'master')
    assert not any(master.values()), 'these numbers are below every master threshold'


# ----------------------------------------------------------------- packaging

def test_package_submission_layout(tmp_path):
    pred = tmp_path / 'pred'
    pred.mkdir()
    for name in ('quiz_037.nii.gz', 'quiz_045.nii.gz'):
        _write_label_map(pred / name, np.zeros((4, 4, 4), dtype=np.uint8))
    write_subtype_outputs({'quiz_037': np.array([1.0, 0.0, 0.0]),
                           'quiz_045': np.array([0.0, 1.0, 0.0])}, str(pred))

    zip_path = tmp_path / 'results.zip'
    package_submission(str(pred), str(zip_path))

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(archive.namelist())
    assert names == ['quiz_037.nii.gz', 'quiz_045.nii.gz', SUBTYPE_CSV]


def test_package_submission_excludes_input_images(tmp_path):
    """_0000 files are inputs, not predictions; they must not end up in the archive."""
    pred = tmp_path / 'pred'
    pred.mkdir()
    _write_label_map(pred / 'quiz_037.nii.gz', np.zeros((4, 4, 4), dtype=np.uint8))
    _write_label_map(pred / 'quiz_037_0000.nii.gz', np.zeros((4, 4, 4), dtype=np.uint8))
    write_subtype_outputs({'quiz_037': np.array([1.0, 0.0, 0.0])}, str(pred))

    zip_path = tmp_path / 'results.zip'
    package_submission(str(pred), str(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        assert 'quiz_037_0000.nii.gz' not in archive.namelist()


def test_package_submission_accepts_separate_fused_subtype_csv(tmp_path):
    pred = tmp_path / 'segmentations'
    pred.mkdir()
    _write_label_map(pred / 'quiz_037.nii.gz', np.zeros((4, 4, 4), dtype=np.uint8))
    write_subtype_outputs({'quiz_037': np.array([1.0, 0.0, 0.0])}, str(pred))

    fused = tmp_path / 'fused'
    fused.mkdir()
    write_subtype_outputs({'quiz_037': np.array([0.0, 0.0, 1.0])}, str(fused))

    zip_path = tmp_path / 'results.zip'
    package_submission(
        str(pred), str(zip_path), subtype_csv_path=str(fused / SUBTYPE_CSV)
    )
    with zipfile.ZipFile(zip_path) as archive:
        subtype_csv = archive.read(SUBTYPE_CSV).decode()
    assert 'quiz_037.nii.gz,2' in subtype_csv
