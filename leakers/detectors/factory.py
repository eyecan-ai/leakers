from leakers.datasets.alphabet import AlphabetDataset
from leakers.detectors.simple import LeakersDetector
from leakers.nn.modules.base import LeakerModule
from leakers.trainers.factory import LeakersInferenceModuleFactory


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
