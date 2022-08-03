from leakers.datasets.alphabet import AlphabetDataset
from leakers.detectors.runes import RunesDetector
from leakers.detectors.simple import LeakersDetector
from leakers.nn.modules.base import LeakerModule
from leakers.trainers.factory import (
    LeakersInferenceModuleFactory,
    RunesInferenceModuleFactory,
)
from leakers.trainers.utils import MasqueradeByImage


class LeakersDetectorsFactory:
    @classmethod
    def create_from_checkpoint(cls, filename: str, device: str = "cpu"):
        module = LeakersInferenceModuleFactory.create_from_checkpoint(filename, device)
        model: LeakerModule = module.model
        dataset: AlphabetDataset = module.proto_dataset
        detector = LeakersDetector(
            model=model, dataset=dataset, grayscale=model.image_shape()[0] == 1
        )
        return detector


class RunesDetectorsFactory:
    @classmethod
    def create_from_checkpoint(cls, filename: str, device: str = "cpu"):
        module = RunesInferenceModuleFactory.create_from_checkpoint(filename, device)
        model: LeakerModule = module.model.to(device)
        dataset: AlphabetDataset = module.proto_dataset

        detector = RunesDetector(
            model=model,
            dataset=dataset,
            grayscale=model.image_shape()[0] == 1,
            device=device,
            masquerade=module.masquerade,
        )
        return detector
