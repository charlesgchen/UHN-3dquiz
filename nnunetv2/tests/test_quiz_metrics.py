import numpy as np

from nnunetv2.evaluation.quiz_metrics import (
    aggregate_segmentation_metrics,
    classification_metrics,
    dice,
    fbeta,
    segmentation_metrics_single_case,
)

VAL_SUPPORT = (9, 15, 12)  # subtype counts in the provided validation split


def _labels_from_support(support):
    return np.concatenate([np.full(n, i) for i, n in enumerate(support)])


def _onehot_probs(labels, num_classes=3):
    return np.eye(num_classes)[labels]


def test_dice_perfect_and_disjoint():
    ref = np.zeros((8, 8, 8), dtype=bool)
    ref[2:6, 2:6, 2:6] = True
    assert dice(ref.copy(), ref) == 1.0

    disjoint = np.zeros_like(ref)
    disjoint[0, 0, 0] = True
    assert dice(disjoint, ref) == 0.0


def test_dice_half_overlap():
    ref = np.zeros(100, dtype=bool)
    ref[:50] = True
    pred = np.zeros(100, dtype=bool)
    pred[25:75] = True
    # |A n B| = 25, |A| = |B| = 50 -> 2*25 / 100 = 0.5
    assert np.isclose(dice(pred, ref), 0.5)


def test_dice_undefined_when_both_empty():
    empty = np.zeros(10, dtype=bool)
    assert np.isnan(dice(empty, empty))


def test_empty_prediction_against_nonempty_reference_is_zero_not_nan():
    """A model that predicts no lesion must score 0, not be silently dropped from the average."""
    ref = np.zeros(10, dtype=bool)
    ref[:3] = True
    assert dice(np.zeros(10, dtype=bool), ref) == 0.0


def test_fbeta_weights_sensitivity_above_dice():
    """
    With beta=2, missing voxels is penalised more than over-segmenting them.

    The two predictions below are constructed to have *identical* Dice (0.8) against the same
    reference, one purely under-segmenting and one purely over-segmenting:
        under: TP=40, FP=0,  FN=20  ->  2*40 / (2*40 + 20 + 0)  = 0.8
        over:  TP=60, FP=30, FN=0   ->  2*60 / (2*60 + 0  + 30) = 0.8
    """
    ref = np.zeros(200, dtype=bool)
    ref[:60] = True

    under = np.zeros(200, dtype=bool)
    under[:40] = True
    over = np.zeros(200, dtype=bool)
    over[:90] = True

    # Dice cannot tell these two failure modes apart
    assert np.isclose(dice(under, ref), 0.8)
    assert np.isclose(dice(over, ref), 0.8)
    # F2 can: under-segmentation scores strictly worse
    assert np.isclose(fbeta(under, ref, beta=2.0), 200 / 280)
    assert np.isclose(fbeta(over, ref, beta=2.0), 300 / 330)
    assert fbeta(under, ref, beta=2.0) < fbeta(over, ref, beta=2.0)


def test_segmentation_single_case_keys_and_perfect_score():
    ref = np.zeros((6, 6, 6), dtype=np.uint8)
    ref[1:5, 1:5, 1:5] = 1
    ref[2:4, 2:4, 2:4] = 2

    m = segmentation_metrics_single_case(ref.copy(), ref)
    assert set(m) == {'dice_whole_pancreas', 'dice_lesion', 'dice_normal_pancreas', 'f2_lesion'}
    assert all(np.isclose(v, 1.0) for v in m.values())


def test_whole_pancreas_ignores_confusion_between_label_1_and_2():
    """label>0 must be insensitive to swapping pancreas and lesion, unlike the per-class dice."""
    ref = np.zeros((6, 6, 6), dtype=np.uint8)
    ref[1:5, 1:5, 1:5] = 1
    ref[2:4, 2:4, 2:4] = 2

    pred = np.where(ref > 0, 1, 0).astype(np.uint8)  # everything called pancreas, lesion missed
    m = segmentation_metrics_single_case(pred, ref)
    assert np.isclose(m['dice_whole_pancreas'], 1.0)
    assert m['dice_lesion'] == 0.0


def test_aggregate_skips_nan_and_reports_n():
    per_case = [
        {'dice_lesion': 1.0},
        {'dice_lesion': 0.0},
        {'dice_lesion': float('nan')},
    ]
    agg = aggregate_segmentation_metrics(per_case, prefix='')
    assert np.isclose(agg['dice_lesion_mean'], 0.5)
    assert agg['dice_lesion_n'] == 2


def test_perfect_classifier():
    y = _labels_from_support(VAL_SUPPORT)
    m = classification_metrics(y, _onehot_probs(y))
    for key in ('cls/mcc', 'cls/balanced_accuracy', 'cls/macro_f1',
                'cls/auroc_macro_ovr', 'cls/average_precision_macro'):
        assert np.isclose(m[key], 1.0), key


def test_majority_class_classifier_is_caught_by_mcc_and_balanced_accuracy():
    """
    The reason MCC is the primary metric: on the 9/15/12 validation split a model that always predicts
    subtype 1 gets 15/36 = 42% accuracy, but must score ~0 on MCC and 1/3 on balanced accuracy.
    """
    y = _labels_from_support(VAL_SUPPORT)
    probs = np.tile([0.0, 1.0, 0.0], (len(y), 1))
    m = classification_metrics(y, probs)

    assert np.isclose(m['cls/accuracy'], 15 / 36)
    assert np.isclose(m['cls/mcc'], 0.0)
    assert np.isclose(m['cls/balanced_accuracy'], 1 / 3)
    assert m['cls/macro_f1'] < 0.25
    assert np.isclose(m['cls/subtype0/sensitivity'], 0.0)
    assert np.isclose(m['cls/subtype1/sensitivity'], 1.0)


def test_specificity_is_high_even_for_the_degenerate_classifier():
    """Documents why specificity is the least useful of the per-class metrics here."""
    y = _labels_from_support(VAL_SUPPORT)
    probs = np.tile([0.0, 1.0, 0.0], (len(y), 1))
    m = classification_metrics(y, probs)
    # never predicts subtype0, so every true negative is correct
    assert np.isclose(m['cls/subtype0/specificity'], 1.0)


def test_per_class_support_matches_split():
    y = _labels_from_support(VAL_SUPPORT)
    m = classification_metrics(y, _onehot_probs(y))
    assert (m['cls/subtype0/support'], m['cls/subtype1/support'], m['cls/subtype2/support']) == VAL_SUPPORT


def test_threshold_free_metrics_are_nan_when_a_class_is_absent():
    """A training batch need not contain all 3 classes; that must not raise."""
    y = np.array([0, 0, 1, 1])
    probs = _onehot_probs(y)
    m = classification_metrics(y, probs)
    assert np.isnan(m['cls/auroc_macro_ovr'])
    assert np.isnan(m['cls/average_precision_macro'])
    assert np.isclose(m['cls/mcc'], 1.0)  # counting metrics still work


def test_auroc_rewards_ranking_even_when_argmax_is_wrong():
    """Threshold-free metrics see calibration that macro-F1 cannot."""
    y = np.array([0, 1, 2, 0, 1, 2])
    # argmax is always class 1, but class 0 and 2 probabilities still rank their own cases higher
    probs = np.array([
        [0.40, 0.45, 0.15],
        [0.20, 0.60, 0.20],
        [0.15, 0.45, 0.40],
        [0.35, 0.50, 0.15],
        [0.20, 0.70, 0.10],
        [0.10, 0.55, 0.35],
    ])
    m = classification_metrics(y, probs)
    assert np.isclose(m['cls/balanced_accuracy'], 1 / 3)
    assert m['cls/auroc_macro_ovr'] > 0.8


def test_probability_rows_need_not_be_argmax_ties_free():
    y = np.array([0, 1, 2])
    probs = np.full((3, 3), 1 / 3)
    m = classification_metrics(y, probs)
    assert np.isfinite(m['cls/mcc'])
    assert np.isclose(m['cls/auroc_macro_ovr'], 0.5)
