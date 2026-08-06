import inspect

from nnunetv2.training.dataloading.lesion_centered_loader import LesionCenteredDataLoader
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLesionCenteredSubtypeHeadAdamW import (
    nnUNetTrainerLesionCenteredSubtypeHeadAdamW,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSubtypeHeadAdamW import (
    nnUNetTrainerSubtypeHeadAdamW,
)


def test_lesion_centered_attention_keeps_successful_head_policy():
    cls = nnUNetTrainerLesionCenteredSubtypeHeadAdamW
    assert issubclass(cls, nnUNetTrainerSubtypeHeadAdamW)
    assert cls.head_initial_lr == 1e-4
    assert cls.cls_selection_smoothing == 0.8
    assert cls.lesion_center_label == 2


def test_constructor_keeps_nnunet_checkpoint_signature():
    parameters = inspect.signature(nnUNetTrainerLesionCenteredSubtypeHeadAdamW.__init__).parameters
    assert tuple(parameters) == ('self', 'plans', 'configuration', 'fold', 'dataset_json', 'device')


def test_trainer_uses_lesion_centered_loader_implementation():
    names = nnUNetTrainerLesionCenteredSubtypeHeadAdamW.get_dataloaders.__code__.co_names
    assert 'LesionCenteredDataLoader' in names
    assert issubclass(LesionCenteredDataLoader, object)
