"""
Evaluate multi-task predictions against the held-out validation split, and package a submission.

Segmentation metrics are computed per case against labelsVal, classification metrics against the
subtype labels in subtype_labels.json. Everything goes through nnunetv2/evaluation/quiz_metrics.py so
the numbers in the report and the numbers logged to W&B come from one implementation.
"""

import argparse
import csv
import os
import zipfile
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

from nnunetv2.evaluation.quiz_metrics import (
    aggregate_segmentation_metrics,
    classification_metrics,
    confusion_matrix_counts,
    format_report,
    segmentation_metrics_single_case,
)
from nnunetv2.inference.predict_quiz import PROBABILITIES_JSON, SUBTYPE_CSV


def _read_label_map(path: str) -> np.ndarray:
    return np.round(sitk.GetArrayFromImage(sitk.ReadImage(path))).astype(np.uint8)


def evaluate_segmentation(prediction_folder: str, reference_folder: str,
                          file_ending: str = '.nii.gz') -> Dict[str, float]:
    reference_files = sorted(f for f in os.listdir(reference_folder) if f.endswith(file_ending))
    if not reference_files:
        raise RuntimeError(f'no reference segmentations found in {reference_folder}')

    per_case = []
    missing = []
    for reference_file in reference_files:
        prediction_path = join(prediction_folder, reference_file)
        if not os.path.isfile(prediction_path):
            missing.append(reference_file)
            continue
        reference = _read_label_map(join(reference_folder, reference_file))
        prediction = _read_label_map(prediction_path)
        per_case.append(segmentation_metrics_single_case(prediction, reference))

    if missing:
        raise RuntimeError(f'{len(missing)} reference case(s) have no prediction, e.g. {missing[:3]}. '
                           f'Refusing to report a score over a subset.')
    return aggregate_segmentation_metrics(per_case)


def evaluate_classification(prediction_folder: str, subtype_labels: Dict[str, int],
                            case_identifiers: List[str],
                            file_ending: str = '.nii.gz') -> Dict[str, float]:
    """
    Uses the full probability vectors when predict_quiz wrote them (needed for AUROC / average
    precision), otherwise falls back to the hard labels in subtype_results.csv.
    """
    probabilities_path = join(prediction_folder, PROBABILITIES_JSON)
    if os.path.isfile(probabilities_path):
        raw = load_json(probabilities_path)
        probabilities = np.array([raw[c] for c in case_identifiers], dtype=float)
    else:
        csv_path = join(prediction_folder, SUBTYPE_CSV)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'neither {PROBABILITIES_JSON} nor {SUBTYPE_CSV} found in '
                                    f'{prediction_folder}')
        hard = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                hard[row['Names'][:-len(file_ending)]] = int(row['Subtype'])
        probabilities = np.eye(3)[[hard[c] for c in case_identifiers]]
        print(f'WARNING: {PROBABILITIES_JSON} not found, falling back to hard labels from '
              f'{SUBTYPE_CSV}. AUROC and average precision will be degenerate.')

    targets = np.array([subtype_labels[c] for c in case_identifiers], dtype=int)
    return classification_metrics(targets, probabilities, num_classes=3, prefix='cls/')


def evaluate_quiz_validation(prediction_folder: str, raw_dataset_folder: str,
                             output_file: Optional[str] = None) -> Dict[str, float]:
    """Full held-out evaluation: segmentation against labelsVal, classification against subtype labels."""
    reference_folder = join(raw_dataset_folder, 'labelsVal')
    subtype_labels = load_json(join(raw_dataset_folder, 'subtype_labels.json'))['validation']
    case_identifiers = sorted(subtype_labels)

    segmentation = evaluate_segmentation(prediction_folder, reference_folder)
    classification = evaluate_classification(prediction_folder, subtype_labels, case_identifiers)

    metrics = {**segmentation, **classification}
    targets = np.array([subtype_labels[c] for c in case_identifiers])
    predictions = []
    raw = load_json(join(prediction_folder, PROBABILITIES_JSON)) \
        if os.path.isfile(join(prediction_folder, PROBABILITIES_JSON)) else None
    if raw is not None:
        predictions = [int(np.argmax(raw[c])) for c in case_identifiers]
        metrics['cls/confusion_matrix'] = confusion_matrix_counts(targets, np.array(predictions)).tolist()

    if output_file:
        save_json(metrics, output_file)
    return metrics


def check_quiz_targets(metrics: Dict[str, float], track: str = 'undergraduate') -> Dict[str, bool]:
    """Compare against the acceptance thresholds in the quiz brief."""
    thresholds = {
        'undergraduate': {'seg/dice_whole_pancreas_mean': 0.90, 'seg/dice_lesion_mean': 0.27,
                          'cls/macro_f1': 0.60},
        'master': {'seg/dice_whole_pancreas_mean': 0.91, 'seg/dice_lesion_mean': 0.31,
                   'cls/macro_f1': 0.70},
    }[track]
    return {k: bool(metrics.get(k, float('nan')) >= v) for k, v in thresholds.items()}


def package_submission(prediction_folder: str, zip_path: str, file_ending: str = '.nii.gz'):
    """
    Zip the test predictions in the layout the quiz asks for: the segmentations plus
    subtype_results.csv, flat at the archive root.
    """
    segmentations = sorted(f for f in os.listdir(prediction_folder)
                           if f.endswith(file_ending) and not f.endswith('_0000' + file_ending))
    csv_path = join(prediction_folder, SUBTYPE_CSV)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'{SUBTYPE_CSV} not found in {prediction_folder}')
    if not segmentations:
        raise RuntimeError(f'no segmentations found in {prediction_folder}')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in segmentations:
            archive.write(join(prediction_folder, name), arcname=name)
        archive.write(csv_path, arcname=SUBTYPE_CSV)
    print(f'wrote {zip_path} with {len(segmentations)} segmentations + {SUBTYPE_CSV}')


def evaluate_quiz_entry():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', type=str, required=True, help='folder with predictions')
    parser.add_argument('-d', type=str, required=True,
                        help='raw dataset folder (…/nnUNet_raw/Dataset001_PancreasQuiz)')
    parser.add_argument('-o', type=str, default=None, help='where to write summary json')
    parser.add_argument('--track', type=str, default='undergraduate',
                        choices=['undergraduate', 'master'])
    args = parser.parse_args()

    metrics = evaluate_quiz_validation(args.i, args.d, args.o)
    print(format_report({k: v for k, v in metrics.items() if not isinstance(v, list)},
                        'Held-out validation (36 cases)'))
    print()
    for name, passed in check_quiz_targets(metrics, args.track).items():
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')


if __name__ == '__main__':
    evaluate_quiz_entry()
