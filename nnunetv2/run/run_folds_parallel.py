"""
Launch several nnU-Net folds concurrently, one per GPU.

This is NOT the same thing as `nnUNetv2_train ... -num_gpus N`:

    -num_gpus N          DDP. ONE fold, its batch split across N GPUs. Finishes one fold faster and
                         changes the effective batch size (nnU-Net compensates by adjusting the
                         per-worker oversampling).
    this script          N independent folds, one per GPU, in parallel. Each fold is an ordinary
                         single-GPU training with the batch size the plans specify, so results are
                         identical to running the folds one after another - just wall-clock faster.

For cross-validation you almost always want the second: the folds are independent by construction, so
running them concurrently costs nothing statistically.

GPU selection uses CUDA_VISIBLE_DEVICES per child process, which is what nnU-Net expects - the
-device flag must not be used to pick a GPU index (see the help text on nnUNetv2_train -device).

W&B runs are grouped so the folds show up as one experiment: WANDB_RUN_GROUP and WANDB_NAME are read
natively by wandb, so no logger changes are needed.

Example
-------
    python -m nnunetv2.run.run_folds_parallel -d 1 -c 3d_fullres -f 0 1 2 3 4 -g 0 1 \
        -tr nnUNetTrainerMultiTaskSubtype -p nnUNetResEncUNetMPlans

With 5 folds and 2 GPUs this runs folds 0,1 first, then 2,3, then 4 - each GPU takes the next queued
fold as it frees up.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import List, Optional


def _run_one_fold(dataset_id: str, configuration: str, fold: int, gpu: int, trainer: str,
                  plans: str, run_group: str, extra_args: List[str], log_folder: Optional[str]) -> int:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # wandb reads these directly; grouping keeps the 5 folds together as one experiment
    env['WANDB_RUN_GROUP'] = run_group
    env['WANDB_NAME'] = f'{run_group}_fold{fold}'

    command = ['nnUNetv2_train', str(dataset_id), configuration, str(fold),
               '-tr', trainer, '-p', plans] + extra_args

    print(f'[gpu {gpu}] fold {fold}: {" ".join(command)}', flush=True)

    log_handle = None
    if log_folder:
        os.makedirs(log_folder, exist_ok=True)
        log_handle = open(os.path.join(log_folder, f'fold_{fold}_gpu_{gpu}.log'), 'w')

    try:
        completed = subprocess.run(command, env=env,
                                   stdout=log_handle or None,
                                   stderr=subprocess.STDOUT if log_handle else None)
        return completed.returncode
    finally:
        if log_handle:
            log_handle.close()


def run_folds_parallel(dataset_id: str, configuration: str, folds: List[int], gpus: List[int],
                       trainer: str, plans: str, run_group: Optional[str] = None,
                       extra_args: Optional[List[str]] = None,
                       log_folder: Optional[str] = None) -> int:
    """
    Runs `folds` across `gpus`, one fold per GPU at a time. Returns 0 if every fold succeeded.

    Folds are handed out from a shared queue rather than pre-assigned, so a GPU that finishes early
    picks up the next fold instead of idling.
    """
    extra_args = extra_args or []
    run_group = run_group or f'd{dataset_id}_{configuration}_{datetime.now():%Y%m%d-%H%M%S}'

    pending: "Queue[int]" = Queue()
    for fold in folds:
        pending.put(fold)

    results = {}

    def worker(gpu: int):
        while True:
            try:
                fold = pending.get_nowait()
            except Exception:
                return
            try:
                results[fold] = _run_one_fold(dataset_id, configuration, fold, gpu, trainer, plans,
                                              run_group, extra_args, log_folder)
            finally:
                pending.task_done()

    started = time.time()
    threads = [Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(f'\nW&B group: {run_group}')
    print(f'elapsed: {(time.time() - started) / 60:.1f} min')
    failed = {fold: code for fold, code in results.items() if code != 0}
    for fold in sorted(results):
        print(f'  fold {fold}: {"ok" if results[fold] == 0 else f"FAILED (exit {results[fold]})"}')
    if failed:
        print(f'\n{len(failed)} fold(s) failed: {sorted(failed)}')
        return 1
    return 0


def run_folds_parallel_entry():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', type=str, required=True, help='dataset name or id')
    parser.add_argument('-c', type=str, default='3d_fullres', help='configuration')
    parser.add_argument('-f', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='folds to run')
    parser.add_argument('-g', type=int, nargs='+', default=[0], help='GPU indices to spread them over')
    parser.add_argument('-tr', type=str, default='nnUNetTrainerMultiTaskSubtype', help='trainer class')
    parser.add_argument('-p', type=str, default='nnUNetResEncUNetMPlans', help='plans identifier')
    parser.add_argument('--group', type=str, default=None, help='W&B run group (default: auto)')
    parser.add_argument('--log_folder', type=str, default=None,
                        help='write each fold\'s stdout to a file here instead of the terminal')
    parser.add_argument('--extra', type=str, nargs=argparse.REMAINDER, default=[],
                        help='everything after this is forwarded verbatim to nnUNetv2_train')
    args = parser.parse_args()

    sys.exit(run_folds_parallel(args.d, args.c, args.f, args.g, args.tr, args.p,
                                run_group=args.group, extra_args=args.extra,
                                log_folder=args.log_folder))


if __name__ == '__main__':
    run_folds_parallel_entry()
