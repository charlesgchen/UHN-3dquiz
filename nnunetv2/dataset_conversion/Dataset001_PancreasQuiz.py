"""
Converts the UHN 3D quiz data (train/ validation/ test/ folders at the repo root) into nnU-Net v2 format.

Source layout:
    train/subtype{0,1,2}/quiz_{subtype}_{id}_0000.nii.gz   (CT image)
    train/subtype{0,1,2}/quiz_{subtype}_{id}.nii.gz        (segmentation, labels 0/1/2)
    validation/subtype{0,1,2}/...                          (same structure)
    test/quiz_{id}_0000.nii.gz                             (CT image, no segmentation)

Target layout (nnUNet_raw/Dataset001_PancreasQuiz):
    imagesTr/  labelsTr/   <- train split only. This is the ONLY data nnU-Net plans/preprocesses/trains on.
    imagesVal/ labelsVal/  <- provided validation split, held out. nnU-Net addresses its folders by the
                              fixed names imagesTr/labelsTr/imagesTs, so these are invisible to the
                              framework and can never leak into fingerprinting or training.
    imagesTs/              <- provided test images.
    subtype_labels.json    <- case identifier -> subtype (0/1/2), for the classification head.

The segmentations are stored as float with values {0, 1, 1.000015, 2}. They are rounded and written as
uint8 so that the label maps are exact integers, as nnU-Net requires.

This script deliberately depends only on the standard library, numpy and SimpleITK so that it can be run
before `pip install -e .` has made the nnunetv2 package (and batchgenerators) importable.
"""

import json
import multiprocessing
import os
import shutil
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

SUBTYPE_FOLDERS = ('subtype0', 'subtype1', 'subtype2')
FILE_ENDING = '.nii.gz'
CHANNEL_SUFFIX = '_0000' + FILE_ENDING


def list_images(folder: str) -> List[str]:
    """Sorted file names in folder that are nnU-Net channel-0 images."""
    return sorted(i for i in os.listdir(folder) if i.endswith(CHANNEL_SUFFIX))


def collect_split(split_folder: str) -> List[Tuple[str, int, str, str]]:
    """Returns (case_identifier, subtype, image_path, label_path) for every case in a split folder."""
    cases = []
    for subtype_folder in SUBTYPE_FOLDERS:
        subtype = int(subtype_folder[len('subtype'):])
        folder = join(split_folder, subtype_folder)
        for image_file in list_images(folder):
            case_identifier = image_file[:-len(CHANNEL_SUFFIX)]
            label_path = join(folder, case_identifier + FILE_ENDING)
            assert os.path.isfile(label_path), f'no segmentation for case {case_identifier} (expected {label_path})'
            cases.append((case_identifier, subtype, join(folder, image_file), label_path))
    return cases


def write_json(obj, target: str) -> None:
    with open(target, 'w') as f:
        json.dump(obj, f, indent=4)


def convert_segmentation(source: str, target: str) -> None:
    """Round the float segmentation to exact integers and save as uint8, preserving all header info."""
    itk_image = sitk.ReadImage(source)
    seg = np.round(sitk.GetArrayFromImage(itk_image)).astype(np.uint8)
    itk_out = sitk.GetImageFromArray(seg)
    itk_out.CopyInformation(itk_image)
    sitk.WriteImage(itk_out, target, useCompression=True)


def convert_dataset(source_root: str, raw_data_folder: str, num_processes: int = 8) -> None:
    dataset_folder = join(raw_data_folder, 'Dataset001_PancreasQuiz')
    for sub in ('imagesTr', 'labelsTr', 'imagesVal', 'labelsVal', 'imagesTs'):
        os.makedirs(join(dataset_folder, sub), exist_ok=True)

    subtype_labels: Dict[str, Dict[str, int]] = {}
    seg_jobs = []

    for split, images_folder, labels_folder in (('train', 'imagesTr', 'labelsTr'),
                                                ('validation', 'imagesVal', 'labelsVal')):
        cases = collect_split(join(source_root, split))
        subtype_labels[split] = {}
        for case_identifier, subtype, image_path, label_path in cases:
            assert case_identifier not in subtype_labels[split], f'duplicate case identifier {case_identifier}'
            shutil.copy(image_path, join(dataset_folder, images_folder, case_identifier + CHANNEL_SUFFIX))
            seg_jobs.append((label_path, join(dataset_folder, labels_folder, case_identifier + FILE_ENDING)))
            subtype_labels[split][case_identifier] = subtype
        print(f'{split}: {len(cases)} cases -> {images_folder}/{labels_folder}')

    test_images = list_images(join(source_root, 'test'))
    for image_file in test_images:
        shutil.copy(join(source_root, 'test', image_file), join(dataset_folder, 'imagesTs', image_file))
    print(f'test: {len(test_images)} images -> imagesTs')

    with multiprocessing.get_context('spawn').Pool(num_processes) as p:
        p.starmap(convert_segmentation, seg_jobs)

    write_json(subtype_labels, join(dataset_folder, 'subtype_labels.json'))

    # equivalent to nnunetv2.dataset_conversion.generate_dataset_json, written directly to keep this
    # script importable without the nnunetv2 package being installed
    dataset_json = {
        'channel_names': {'0': 'CT'},
        'labels': {'background': 0, 'pancreas': 1, 'lesion': 2},
        'numTraining': len(subtype_labels['train']),
        'file_ending': FILE_ENDING,
        'name': 'PancreasQuiz',
        'description': 'Pancreas CT with pancreas/lesion segmentation and a per-case subtype label (0/1/2). '
                       'Only imagesTr/labelsTr are visible to nnU-Net; the provided validation split lives '
                       'in imagesVal/labelsVal and is held out.',
        'licence': 'see source dataset',
        'converted_by': 'Dataset001_PancreasQuiz.py',
    }
    write_json(dataset_json, join(dataset_folder, 'dataset.json'))
    print(f'done -> {dataset_folder}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='folder containing train/, validation/ and test/')
    parser.add_argument('-o', type=str, required=True, help='nnUNet_raw folder')
    parser.add_argument('-np', type=int, default=8, help='number of processes for segmentation conversion')
    args = parser.parse_args()
    convert_dataset(args.i, args.o, args.np)
