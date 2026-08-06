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
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

SUBTYPE_CSV = 'subtype_results.csv'
PROBABILITIES_CSV = 'subtype_probabilities.csv'
PROBABILITIES_JSON = 'subtype_probabilities.json'


def resolve_mirroring_axes(stored_axes: Optional[Tuple[int, ...]],
                           requested_axes: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    """Validate an explicit TTA-axis subset against the axes stored in the checkpoint."""
    if requested_axes is None:
        return stored_axes
    requested_axes = tuple(sorted(set(int(axis) for axis in requested_axes)))
    if not requested_axes:
        raise ValueError('an explicit TTA axis subset cannot be empty')
    if any(axis not in (0, 1, 2) for axis in requested_axes):
        raise ValueError(f'TTA axes must be drawn from (0, 1, 2), got {requested_axes}')
    if stored_axes is None:
        raise ValueError('the checkpoint does not permit mirroring TTA')
    stored_axes = tuple(int(axis) for axis in stored_axes)
    if not set(requested_axes).issubset(stored_axes):
        raise ValueError(
            f'requested TTA axes {requested_axes} are not a subset of checkpoint axes {stored_axes}')
    return requested_axes


def gamma_intensity_view(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply deterministic per-channel gamma while retaining spatial mean and variance.

    This mirrors the statistics-retaining gamma augmentation used during nnU-Net training. It is
    deliberately applied only to classification TTA views; the submitted segmentation remains the
    ordinary spatial-TTA prediction.
    """
    if gamma <= 0:
        raise ValueError(f'gamma must be positive, got {gamma}')
    spatial_axes = tuple(range(2, x.ndim))
    minimum = x.amin(dim=spatial_axes, keepdim=True)
    maximum = x.amax(dim=spatial_axes, keepdim=True)
    value_range = maximum - minimum
    normalized = (x - minimum) / value_range.clamp_min(1e-6)
    transformed = normalized.clamp(0, 1).pow(gamma) * value_range + minimum

    original_mean = x.mean(dim=spatial_axes, keepdim=True)
    original_std = x.std(dim=spatial_axes, keepdim=True, correction=0)
    transformed_mean = transformed.mean(dim=spatial_axes, keepdim=True)
    transformed_std = transformed.std(dim=spatial_axes, keepdim=True, correction=0)
    transformed = (
        (transformed - transformed_mean)
        * (original_std / transformed_std.clamp_min(1e-6))
        + original_mean
    )
    constant = value_range <= 1e-6
    return torch.where(constant, x, transformed)


class MultiTaskPredictor(nnUNetPredictor):
    """
    nnUNetPredictor for a network whose forward returns (segmentation, classification_logits).

    Only _internal_maybe_mirror_and_predict is overridden: it unpacks the tuple, accumulates the
    classification probabilities for the case currently being predicted, and hands the segmentation
    back to the unmodified sliding-window machinery.
    """

    def __init__(self, *args, foreground_weighted_pooling: bool = True,
                 classification_patch_pooling: str = 'weighted_mean',
                 classification_gamma_tta: Tuple[float, ...] = (),
                 save_classification_embeddings: bool = False,
                 save_classification_patch_embeddings: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.foreground_weighted_pooling = foreground_weighted_pooling
        if classification_patch_pooling not in {'weighted_mean', 'top1', 'top3'}:
            raise ValueError(f'unknown classification_patch_pooling: {classification_patch_pooling}')
        self.classification_patch_pooling = classification_patch_pooling
        self.classification_gamma_tta = tuple(float(gamma) for gamma in classification_gamma_tta)
        if any(gamma <= 0 for gamma in self.classification_gamma_tta):
            raise ValueError('classification gamma TTA values must be positive')
        self.save_classification_embeddings = save_classification_embeddings
        self.save_classification_patch_embeddings = save_classification_patch_embeddings
        self._reset_classification_accumulator()

    def _reset_classification_accumulator(self):
        self._cls_weighted_sum: Optional[np.ndarray] = None
        self._cls_uniform_sum: Optional[np.ndarray] = None
        self._cls_weight_total: float = 0.0
        self._cls_patch_count: int = 0
        self._cls_member_accumulators: Dict[int, dict] = {}
        self._active_classification_member: Optional[int] = None

    def _accumulate_classification(self, logits: torch.Tensor, segmentation: torch.Tensor):
        probabilities = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()

        if self.foreground_weighted_pooling:
            seg_probabilities = torch.softmax(segmentation.float(), dim=1)
            network = getattr(self, 'network', None)
            raw_network = network._orig_mod if hasattr(network, '_orig_mod') else network
            pooling_label = getattr(raw_network, 'classification_pooling_label', None)
            if pooling_label is None:
                # Default multi-task head: use the fraction predicted as pancreas or lesion.
                evidence = seg_probabilities[:, 1:].sum(dim=1)
            else:
                pooling_label = int(pooling_label)
                if not 0 <= pooling_label < seg_probabilities.shape[1]:
                    raise ValueError(
                        f'classification_pooling_label {pooling_label} is outside segmentation channels '
                        f'[0, {seg_probabilities.shape[1]})')
                # ROI heads are trained on lesion-local evidence, so only lesion-bearing patches vote strongly.
                evidence = seg_probabilities[:, pooling_label]
            weights = evidence.flatten(1).mean(dim=1).detach().cpu().numpy()
        else:
            weights = np.ones(probabilities.shape[0], dtype=np.float64)

        weighted = (probabilities * weights[:, None]).sum(axis=0)
        uniform = probabilities.sum(axis=0)

        self._cls_weighted_sum = weighted if self._cls_weighted_sum is None else self._cls_weighted_sum + weighted
        self._cls_uniform_sum = uniform if self._cls_uniform_sum is None else self._cls_uniform_sum + uniform
        self._cls_weight_total += float(weights.sum())
        self._cls_patch_count += probabilities.shape[0]

        # Patch evidence is pooled within each fold first. Otherwise a model that predicts a larger
        # foreground volume receives more weight in the ensemble than a model that predicts a smaller
        # volume. Folds are independent, equally weighted estimators and must each contribute one vote.
        member = self._active_classification_member
        member = 0 if member is None else int(member)
        state = self._cls_member_accumulators.get(member)
        if state is None:
            state = {
                'weighted_sum': np.zeros_like(weighted),
                'uniform_sum': np.zeros_like(uniform),
                'weight_total': 0.0,
                'patch_count': 0,
                'candidates': [],
                'embedding_weighted_sum': None,
                'embedding_uniform_sum': None,
                'patch_embeddings': [],
                'patch_embedding_weights': [],
            }
            self._cls_member_accumulators[member] = state
        state['weighted_sum'] += weighted
        state['uniform_sum'] += uniform
        state['weight_total'] += float(weights.sum())
        state['patch_count'] += probabilities.shape[0]
        state['candidates'].extend(
            (float(weight), probability.copy())
            for weight, probability in zip(weights, probabilities)
        )
        if (getattr(self, 'save_classification_embeddings', False) or
                getattr(self, 'save_classification_patch_embeddings', False)):
            network = getattr(self, 'network', None)
            raw_network = network._orig_mod if hasattr(network, '_orig_mod') else network
            embedding = getattr(raw_network, 'last_classification_embedding', None)
            if embedding is None:
                raise RuntimeError('network did not expose a classification embedding')
            embedding = embedding.detach().float().cpu().numpy()
            weighted_embedding = (embedding * weights[:, None]).sum(axis=0)
            uniform_embedding = embedding.sum(axis=0)
            if state['embedding_weighted_sum'] is None:
                state['embedding_weighted_sum'] = weighted_embedding
                state['embedding_uniform_sum'] = uniform_embedding
            else:
                state['embedding_weighted_sum'] += weighted_embedding
                state['embedding_uniform_sum'] += uniform_embedding
            if getattr(self, 'save_classification_patch_embeddings', False):
                state['patch_embeddings'].append(embedding.copy())
                state['patch_embedding_weights'].append(weights.copy())

    def get_case_classification(self) -> np.ndarray:
        """Aggregated subtype probabilities for the case just predicted."""
        if self._cls_patch_count == 0:
            raise RuntimeError('no classification logits were accumulated; predict a case first')
        member_probabilities = []
        for state in self._cls_member_accumulators.values():
            patch_pooling = getattr(self, 'classification_patch_pooling', 'weighted_mean')
            if patch_pooling in {'top1', 'top3'}:
                # Lesion-centred heads see one lesion-bearing crop at training time. Selecting the
                # strongest lesion-evidence window tests the matching inference rule without using
                # labels, morphology, coordinates, or any trainable segmentation parameters.
                top_k = 1 if patch_pooling == 'top1' else 3
                selected = sorted(state['candidates'], key=lambda item: item[0], reverse=True)[:top_k]
                selected_weights = np.asarray([item[0] for item in selected], dtype=np.float64)
                selected_probabilities = np.asarray([item[1] for item in selected])
                if selected_weights.sum() > 1e-6:
                    probabilities = np.average(selected_probabilities, axis=0, weights=selected_weights)
                else:
                    probabilities = selected_probabilities.mean(axis=0)
            # A weight total this small means this fold saw no meaningful foreground. Fall back to
            # uniform patch pooling for that fold, while still giving the fold one ensemble vote.
            elif state['weight_total'] > 1e-6:
                probabilities = state['weighted_sum'] / state['weight_total']
            else:
                probabilities = state['uniform_sum'] / state['patch_count']
            member_probabilities.append(probabilities / probabilities.sum())
        probabilities = np.mean(member_probabilities, axis=0)
        return probabilities / probabilities.sum()

    def get_case_classification_embeddings(self) -> np.ndarray:
        """Return one lesion/foreground-weighted frozen encoder embedding per ensemble member."""
        embeddings = []
        # Keep the saved row index equal to the requested nnU-Net fold/member index. Dict insertion
        # order normally already matches prediction order, but explicit sorting makes aligned MLP
        # checkpoints robust to callers that request folds in a different order.
        for member in sorted(self._cls_member_accumulators):
            state = self._cls_member_accumulators[member]
            if state['embedding_weighted_sum'] is None:
                raise RuntimeError('classification embeddings were not captured')
            if state['weight_total'] > 1e-6:
                embedding = state['embedding_weighted_sum'] / state['weight_total']
            else:
                embedding = state['embedding_uniform_sum'] / state['patch_count']
            embeddings.append(embedding)
        return np.stack(embeddings)

    def get_case_classification_patch_embeddings(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return every frozen patch embedding and its segmentation-derived evidence per fold."""
        members = []
        for member in sorted(self._cls_member_accumulators):
            state = self._cls_member_accumulators[member]
            if not state['patch_embeddings']:
                raise RuntimeError('classification patch embeddings were not captured')
            embeddings = np.concatenate(state['patch_embeddings'], axis=0)
            weights = np.concatenate(state['patch_embedding_weights'], axis=0)
            if len(embeddings) != len(weights):
                raise RuntimeError('patch embedding and evidence counts differ')
            members.append((embeddings, weights))
        return members

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """Mirror nnU-Net's fold ensemble while exposing the active fold to the classifier pool."""
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(default_num_processes, n_threads))
        prediction = None
        try:
            for member, parameters in enumerate(self.list_of_parameters):
                self._active_classification_member = member
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(parameters)
                else:
                    self.network._orig_mod.load_state_dict(parameters)

                member_prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                prediction = member_prediction if prediction is None else prediction + member_prediction

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)
            if self.verbose:
                print('Prediction done')
            return prediction
        finally:
            self._active_classification_member = None
            torch.set_num_threads(n_threads)

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
        # Intensity views provide extra classification votes but intentionally do not alter the
        # segmentation prediction. Applying only the unmirrored gamma views limits 8x spatial TTA
        # plus two gammas to ten forwards rather than multiplying them into 24 forwards.
        for gamma in getattr(self, 'classification_gamma_tta', ()):
            gamma_seg, gamma_cls = self.network(gamma_intensity_view(x, gamma))
            self._accumulate_classification(gamma_cls, gamma_seg)
        return prediction


def predict_folder(model_folder: str, input_folder: str, output_folder: str,
                   folds: Tuple = (0,), checkpoint_name: str = 'checkpoint_final.pth',
                   tile_step_size: float = 0.5, use_mirroring: bool = True,
                   mirroring_axes: Optional[Tuple[int, ...]] = None,
                   foreground_weighted_pooling: bool = True,
                   classification_patch_pooling: str = 'weighted_mean',
                   classification_gamma_tta: Tuple[float, ...] = (),
                   save_classification_embeddings: bool = False,
                   save_classification_patch_embeddings: bool = False,
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
        classification_patch_pooling=classification_patch_pooling,
        classification_gamma_tta=classification_gamma_tta,
        save_classification_embeddings=save_classification_embeddings,
        save_classification_patch_embeddings=save_classification_patch_embeddings,
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds,
                                                   checkpoint_name=checkpoint_name)
    if mirroring_axes is not None and not use_mirroring:
        raise ValueError('mirroring_axes cannot be supplied when mirroring TTA is disabled')
    if use_mirroring:
        predictor.allowed_mirroring_axes = resolve_mirroring_axes(
            predictor.allowed_mirroring_axes, mirroring_axes)
    raw_network = predictor.network._orig_mod if hasattr(predictor.network, '_orig_mod') else predictor.network
    raw_network.capture_classification_embedding = (
        save_classification_embeddings or save_classification_patch_embeddings)

    file_ending = predictor.dataset_json['file_ending']
    case_files = sorted(f for f in os.listdir(input_folder) if f.endswith('_0000' + file_ending))
    if not case_files:
        raise RuntimeError(f'no images matching *_0000{file_ending} found in {input_folder}')

    probabilities: Dict[str, np.ndarray] = {}
    embeddings_dir = join(output_folder, 'classification_embeddings')
    if save_classification_embeddings:
        maybe_mkdir_p(embeddings_dir)
    patch_embeddings_dir = join(output_folder, 'classification_patch_embeddings')
    if save_classification_patch_embeddings:
        maybe_mkdir_p(patch_embeddings_dir)
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
        if save_classification_embeddings:
            np.save(join(embeddings_dir, case_identifier + '.npy'),
                    predictor.get_case_classification_embeddings())
        if save_classification_patch_embeddings:
            arrays = {}
            for member, (patch_embeddings, patch_weights) in enumerate(
                    predictor.get_case_classification_patch_embeddings()):
                arrays[f'embeddings_member_{member}'] = patch_embeddings
                arrays[f'weights_member_{member}'] = patch_weights
            np.savez_compressed(join(patch_embeddings_dir, case_identifier + '.npz'), **arrays)
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
    parser.add_argument('--tta_axes', type=int, nargs='+', choices=(0, 1, 2),
                        help='optional subset of checkpoint mirror axes; one axis gives 2x TTA, '
                             'two axes 4x, and three axes 8x')
    parser.add_argument('--uniform_pooling', action='store_true',
                        help='average patch subtype probabilities uniformly instead of weighting them '
                             'by predicted foreground')
    parser.add_argument('--classification_patch_pooling', default='weighted_mean',
                        choices=('weighted_mean', 'top1', 'top3'),
                        help='reduce patch predictions within each fold; top-k ranks windows by the '
                             'configured segmentation evidence (lesion for lesion-centred heads)')
    parser.add_argument(
        '--classification_gamma_tta', type=float, nargs='+',
        help='additional classification-only statistics-retaining gamma views, for example 0.85 1.15',
    )
    parser.add_argument('--save_classification_embeddings', action='store_true',
                        help='save lesion-weighted frozen bottleneck mean/max embeddings per fold')
    parser.add_argument('--save_classification_patch_embeddings', action='store_true',
                        help='save every frozen bottleneck mean/max patch embedding and lesion weight')
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
        mirroring_axes=None if args.tta_axes is None else tuple(args.tta_axes),
        foreground_weighted_pooling=not args.uniform_pooling,
        classification_patch_pooling=args.classification_patch_pooling,
        classification_gamma_tta=tuple(args.classification_gamma_tta or ()),
        save_classification_embeddings=args.save_classification_embeddings,
        save_classification_patch_embeddings=args.save_classification_patch_embeddings,
        device=torch.device(args.device),
        verbose=args.verbose,
    )


if __name__ == '__main__':
    predict_quiz_entry()
