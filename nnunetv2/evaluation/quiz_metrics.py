"""
Metrics for the pancreas subtype quiz, selected following Metrics Reloaded
(Maier-Hein et al., Nat Methods 21, 195-212, 2024).

Problem fingerprint that drives the selection:
  * Classification: 3 classes, moderate imbalance (train 62/106/84, val 9/15/12), no clinically
    defined misclassification costs, and only 36 validation cases.
  * Segmentation: semantic segmentation of one large structure (pancreas, ~5.7% of the ROI) and one
    very small structure (lesion, median 0.53% of the ROI, smallest 0.001%). Every case contains
    both classes, so the reference is never empty.

CLASSIFICATION
  Counting / confusion-matrix metrics:
    - Matthews correlation coefficient (MCC)  <- primary summary
    - Balanced accuracy (= macro-averaged sensitivity)
    - Macro-averaged F1                       <- mandated by the quiz brief
  Per class:
    - Sensitivity, specificity, F_beta (beta=1 by default)
  Multi-threshold:
    - AUROC, one-vs-rest, macro averaged
    - Average precision, one-vs-rest, macro averaged

  Why MCC as the primary number: with 9/15/12 validation cases, plain accuracy is maximised by a
  degenerate classifier that always predicts the majority class. Metrics Reloaded recommends MCC for
  multi-class problems where all classes matter equally and no cost matrix exists, because it uses
  all four confusion-matrix entries and stays near 0 for such degenerate predictors. Balanced accuracy
  is reported alongside it because it is far easier to interpret, and macro-F1 because the brief
  specifies it as the acceptance threshold.

  Why NOT expected cost: Metrics Reloaded recommends expected cost only when the application supplies
  a cost matrix. The quiz defines no relative cost for confusing subtype 1 with subtype 2, so any
  matrix would be invented, and the resulting number would not be comparable to anything.

  Caveat on specificity: it is included because it was requested, but in a 3-class one-vs-rest setting
  the negative set is 2-3x larger than the positive set, so specificity is high almost by construction
  and is the least discriminative of the per-class metrics here. Prefer sensitivity and F1 per class.

SEGMENTATION
    - DSC for whole pancreas, np.uint8(label > 0)   <- quiz acceptance target
    - DSC for lesion, np.uint8(label == 2)          <- quiz acceptance target
    - DSC for normal pancreas (label == 1)
    - F_beta with beta=2 for the lesion

  DSC is the standard overlap metric and is what the brief thresholds on. F2 is reported for the
  lesion because DSC (= F1) weights false positives and false negatives equally, while for a structure
  with a median size of 0.5% of the image the dominant failure mode is missing it entirely; beta=2
  weights sensitivity 4x higher and exposes that failure. Metrics are computed per case and then
  averaged over cases (macro averaging), not pooled over all voxels, so that a few large cases cannot
  dominate the score.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

DEFAULT_CLASS_NAMES = ('subtype0', 'subtype1', 'subtype2')


def _fbeta_from_counts(tp: float, fp: float, fn: float, beta: float) -> float:
    """F_beta from raw counts. Returns nan when the metric is undefined (empty reference AND prediction)."""
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom == 0:
        return float('nan')
    return float((1 + b2) * tp / denom)


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """DSC between two boolean masks. nan if both are empty (metric undefined, case should be excluded)."""
    return _fbeta_from_counts(
        tp=float(np.sum(prediction & reference)),
        fp=float(np.sum(prediction & ~reference)),
        fn=float(np.sum(~prediction & reference)),
        beta=1.0,
    )


def fbeta(prediction: np.ndarray, reference: np.ndarray, beta: float = 2.0) -> float:
    """F_beta between two boolean masks. beta > 1 weights sensitivity higher."""
    return _fbeta_from_counts(
        tp=float(np.sum(prediction & reference)),
        fp=float(np.sum(prediction & ~reference)),
        fn=float(np.sum(~prediction & reference)),
        beta=beta,
    )


def segmentation_metrics_single_case(prediction: np.ndarray, reference: np.ndarray,
                                     lesion_beta: float = 2.0) -> Dict[str, float]:
    """
    Segmentation metrics for one case.

    Args:
        prediction: integer label map with values in {0, 1, 2}
        reference:  integer label map with values in {0, 1, 2}, same shape as prediction
        lesion_beta: beta for the additional lesion F_beta

    Returns:
        dict of metric name -> value
    """
    assert prediction.shape == reference.shape, \
        f'prediction and reference must have the same shape, got {prediction.shape} and {reference.shape}'

    pred_whole, ref_whole = prediction > 0, reference > 0
    pred_panc, ref_panc = prediction == 1, reference == 1
    pred_lesion, ref_lesion = prediction == 2, reference == 2

    return {
        'dice_whole_pancreas': dice(pred_whole, ref_whole),
        'dice_lesion': dice(pred_lesion, ref_lesion),
        'dice_normal_pancreas': dice(pred_panc, ref_panc),
        f'f{lesion_beta:g}_lesion': fbeta(pred_lesion, ref_lesion, beta=lesion_beta),
    }


def aggregate_segmentation_metrics(per_case: Sequence[Dict[str, float]],
                                   prefix: str = 'seg/') -> Dict[str, float]:
    """
    Macro-average per-case segmentation metrics, ignoring nan (undefined) entries.

    Args:
        per_case: list of dicts as returned by segmentation_metrics_single_case
        prefix: prepended to every returned key

    Returns:
        dict with mean, std and the number of cases actually used for each metric
    """
    if len(per_case) == 0:
        return {}
    out: Dict[str, float] = {}
    for key in per_case[0].keys():
        values = np.array([case[key] for case in per_case], dtype=float)
        valid = values[~np.isnan(values)]
        out[f'{prefix}{key}_mean'] = float(np.mean(valid)) if valid.size else float('nan')
        out[f'{prefix}{key}_std'] = float(np.std(valid)) if valid.size else float('nan')
        out[f'{prefix}{key}_n'] = int(valid.size)
    return out


def classification_metrics(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           num_classes: int = 3,
                           beta: float = 1.0,
                           class_names: Optional[Sequence[str]] = None,
                           prefix: str = 'cls/') -> Dict[str, float]:
    """
    Classification metrics for the subtype task.

    Args:
        y_true: (N,) integer class labels in [0, num_classes)
        y_prob: (N, num_classes) predicted probabilities. Rows should sum to ~1.
        num_classes: number of classes
        beta: beta for the per-class F_beta (1.0 reproduces per-class F1)
        class_names: names used in the metric keys, defaults to subtype0/1/2
        prefix: prepended to every returned key

    Returns:
        flat dict of metric name -> value, safe to hand straight to wandb.log
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    assert y_prob.ndim == 2 and y_prob.shape[1] == num_classes, \
        f'y_prob must be (N, {num_classes}), got {y_prob.shape}'
    assert y_prob.shape[0] == y_true.shape[0], \
        f'y_true has {y_true.shape[0]} entries but y_prob has {y_prob.shape[0]}'
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[:num_classes]

    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out: Dict[str, float] = {
        f'{prefix}mcc': float(matthews_corrcoef(y_true, y_pred)),
        f'{prefix}balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        f'{prefix}macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        f'{prefix}accuracy': float(np.trace(cm) / max(cm.sum(), 1)),
    }

    # per class, one-vs-rest
    for class_index, name in enumerate(class_names):
        tp = float(cm[class_index, class_index])
        fn = float(cm[class_index].sum() - tp)
        fp = float(cm[:, class_index].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        out[f'{prefix}{name}/sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
        out[f'{prefix}{name}/specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
        out[f'{prefix}{name}/f{beta:g}'] = _fbeta_from_counts(tp, fp, fn, beta)
        out[f'{prefix}{name}/support'] = int(tp + fn)

    # multi-threshold metrics. These need every class to appear in y_true, which is not guaranteed
    # for an arbitrary subset (e.g. a single training batch), so they degrade to nan rather than raise.
    present = np.unique(y_true)
    if len(present) == num_classes:
        y_onehot = np.eye(num_classes)[y_true]
        out[f'{prefix}auroc_macro_ovr'] = float(
            roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=labels))
        out[f'{prefix}average_precision_macro'] = float(
            average_precision_score(y_onehot, y_prob, average='macro'))
        for class_index, name in enumerate(class_names):
            out[f'{prefix}{name}/auroc'] = float(roc_auc_score(y_onehot[:, class_index], y_prob[:, class_index]))
            out[f'{prefix}{name}/average_precision'] = float(
                average_precision_score(y_onehot[:, class_index], y_prob[:, class_index]))
    else:
        out[f'{prefix}auroc_macro_ovr'] = float('nan')
        out[f'{prefix}average_precision_macro'] = float('nan')
        for name in class_names:
            out[f'{prefix}{name}/auroc'] = float('nan')
            out[f'{prefix}{name}/average_precision'] = float('nan')

    return out


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Plain confusion matrix, rows = reference, cols = prediction."""
    return confusion_matrix(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(),
                            labels=list(range(num_classes)))


def summarize(classification: Optional[Dict[str, float]] = None,
              segmentation: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge classification and segmentation metric dicts into one flat dict for logging."""
    merged: Dict[str, float] = {}
    for part in (classification, segmentation):
        if part:
            merged.update(part)
    return merged


def format_report(metrics: Dict[str, float], title: str = 'Validation metrics') -> str:
    """Human-readable multi-line rendering of a metric dict, for the training log and the report."""
    lines = [title, '=' * len(title)]
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, np.integer)):
            lines.append(f'  {key:48s} {value:d}')
        else:
            lines.append(f'  {key:48s} {value:.4f}')
    return '\n'.join(lines)
