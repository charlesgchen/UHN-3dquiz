"""
Inference for the multi-task pancreas quiz model: segmentation maps + per-case subtype predictions.

Segmentation reuses nnUNetPredictor unchanged (preprocessing, sliding window, resampling back to the
original geometry, export). The only addition is that classification logits produced along the way are
captured and aggregated into one subtype prediction per case.

Patch aggregation
    Cases do not fit in a single patch: with patch [64,128,192] the 252 training cases need 1 patch
    (94 cases), 2 (83), 4 (55), up to 12. Each patch yields its own subtype logits, so they must be
    combined. We average softmax probabilities across patches (and across mirror augmentations),
    weighted by how much pancreas each patch actually contains, estimated from that same forward
    pass's segmentation output. A patch of pure background carries no subtype evidence and should not
    get an equal vote. If every patch of a case has near-zero foreground the weighting degenerates, so
    we fall back to a uniform average rather than dividing by ~0.

Outputs
    <output_folder>/quiz_XXX.nii.gz    segmentation, labels 0/1/2
    <output_folder>/subtype_results.csv  columns: Names, Subtype   (Names = "quiz_XXX.nii.gz")
    <output_folder>/subtype_probabilities.csv  per-class probabilities, for threshold-free metrics
"""

import argparse
import csv
import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

SUBTYPE_CSV = 'subtype_results.csv'
PROBABILITIES_CSV = 'subtype_probabilities.csv'
PROBABILITIES_JSON = 'subtype_probabilities.json'


class MultiTaskPredictor(nnUNetPredictor):
    """
    nnUNetPredictor for a network whose forward returns (segmentation, classification_logits).

    Only _internal_maybe_mirror_and_predict is overridden: it unpacks the tuple, accumulates the
    classification probabilities for the case currently being predicted, and hands the segmentation
    back to the unmodified sliding-window machinery.
    """

    def __init__(self, *args, foreground_weighted_pooling: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.foreground_weighted_pooling = foreground_weighted_pooling
        self._reset_classification_accumulator()

    def _reset_classification_accumulator(self):
        self._cls_weighted_sum: Optional[np.ndarray] = None
        self._cls_uniform_sum: Optional[np.ndarray] = None
        self._cls_weight_total: float = 0.0
        self._cls_patch_count: int = 0

    def _accumulate_classification(self, logits: torch.Tensor, segmentation: torch.Tensor):
        probabilities = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()

        if self.foreground_weighted_pooling:
            # fraction of the patch the network believes is pancreas or lesion
            seg_probabilities = torch.softmax(segmentation.float(), dim=1)
            foreground = seg_probabilities[:, 1:].sum(dim=1)
            weights = foreground.flatten(1).mean(dim=1).detach().cpu().numpy()
        else:
            weights = np.ones(probabilities.shape[0], dtype=np.float64)

        weighted = (probabilities * weights[:, None]).sum(axis=0)
        uniform = probabilities.sum(axis=0)

        self._cls_weighted_sum = weighted if self._cls_weighted_sum is None else self._cls_weighted_sum + weighted
        self._cls_uniform_sum = uniform if self._cls_uniform_sum is None else self._cls_uniform_sum + uniform
        self._cls_weight_total += float(weights.sum())
        self._cls_patch_count += probabilities.shape[0]

    def get_case_classification(self) -> np.ndarray:
        """Aggregated subtype probabilities for the case just predicted."""
        if self._cls_patch_count == 0:
            raise RuntimeError('no classification logits were accumulated; predict a case first')
        # a weight total this small means no patch saw meaningful foreground -> weighting is noise
        if self._cls_weight_total > 1e-6:
            probabilities = self._cls_weighted_sum / self._cls_weight_total
        else:
            probabilities = self._cls_uniform_sum / self._cls_patch_count
        return probabilities / probabilities.sum()

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None

        prediction, cls_logits = self.network(x)
        self._accumulate_classification(cls_logits, prediction)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                mirrored_seg, mirrored_cls = self.network(torch.flip(x, axes))
                prediction += torch.flip(mirrored_seg, axes)
                # the subtype of a case is invariant to mirroring, so mirrored views are extra votes
                self._accumulate_classification(mirrored_cls, mirrored_seg)
            prediction /= (len(axes_combinations) + 1)
        return prediction


def predict_folder(model_folder: str, input_folder: str, output_folder: str,
                   folds: Tuple = (0,), checkpoint_name: str = 'checkpoint_final.pth',
                   tile_step_size: float = 0.5, use_mirroring: bool = True,
                   foreground_weighted_pooling: bool = True,
                   device: torch.device = torch.device('cuda'),
                   verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Predict every case in input_folder, writing segmentations and the subtype CSVs to output_folder.

    Returns:
        case identifier -> aggregated subtype probability vector
    """
    maybe_mkdir_p(output_folder)

    predictor = MultiTaskPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=False,
        foreground_weighted_pooling=foreground_weighted_pooling,
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds,
                                                   checkpoint_name=checkpoint_name)

    file_ending = predictor.dataset_json['file_ending']
    case_files = sorted(f for f in os.listdir(input_folder) if f.endswith('_0000' + file_ending))
    if not case_files:
        raise RuntimeError(f'no images matching *_0000{file_ending} found in {input_folder}')

    probabilities: Dict[str, np.ndarray] = {}
    for image_file in case_files:
        case_identifier = image_file[:-(len('_0000') + len(file_ending))]
        predictor._reset_classification_accumulator()

        # one case at a time so the classification accumulator maps 1:1 onto a case
        predictor.predict_from_files(
            [[join(input_folder, image_file)]],
            [join(output_folder, case_identifier + file_ending)],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        probabilities[case_identifier] = predictor.get_case_classification()
        print(f'{case_identifier}: subtype {int(np.argmax(probabilities[case_identifier]))} '
              f'({np.round(probabilities[case_identifier], 3).tolist()})')

    write_subtype_outputs(probabilities, output_folder, file_ending)
    return probabilities


def write_subtype_outputs(probabilities: Dict[str, np.ndarray], output_folder: str,
                          file_ending: str = '.nii.gz'):
    """Write subtype_results.csv in the format the quiz asks for, plus the raw probabilities."""
    identifiers = sorted(probabilities)

    with open(join(output_folder, SUBTYPE_CSV), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Names', 'Subtype'])
        for case_identifier in identifiers:
            writer.writerow([case_identifier + file_ending, int(np.argmax(probabilities[case_identifier]))])

    num_classes = len(next(iter(probabilities.values())))
    with open(join(output_folder, PROBABILITIES_CSV), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Names'] + [f'p_subtype{i}' for i in range(num_classes)])
        for case_identifier in identifiers:
            writer.writerow([case_identifier + file_ending] +
                            [float(p) for p in probabilities[case_identifier]])

    save_json({k: [float(p) for p in v] for k, v in probabilities.items()},
              join(output_folder, PROBABILITIES_JSON))


def predict_quiz_entry():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', type=str, required=True,
                        help='trained model folder (…/Dataset001_PancreasQuiz/nnUNetTrainerMultiTaskSubtype__'
                             'nnUNetResEncUNetMPlans__3d_fullres)')
    parser.add_argument('-i', type=str, required=True, help='folder with images (…_0000.nii.gz)')
    parser.add_argument('-o', type=str, required=True, help='output folder')
    parser.add_argument('-f', type=int, nargs='+', default=[0], help='folds to use')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='checkpoint name')
    parser.add_argument('-step_size', type=float, default=0.5, help='sliding window step size')
    parser.add_argument('--disable_tta', action='store_true', help='disable mirroring TTA')
    parser.add_argument('--uniform_pooling', action='store_true',
                        help='average patch subtype probabilities uniformly instead of weighting them '
                             'by predicted foreground')
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    predict_folder(
        model_folder=args.m,
        input_folder=args.i,
        output_folder=args.o,
        folds=tuple(args.f),
        checkpoint_name=args.chk,
        tile_step_size=args.step_size,
        use_mirroring=not args.disable_tta,
        foreground_weighted_pooling=not args.uniform_pooling,
        device=torch.device(args.device),
        verbose=args.verbose,
    )


if __name__ == '__main__':
    predict_quiz_entry()
