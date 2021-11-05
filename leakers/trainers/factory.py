import torch


import rich

from leakers.trainers.modules import LeakersTrainingModule


class LeakersInferenceModuleFactory:
    @classmethod
    def create_from_checkpoint(cls, filename: str, device="cpu"):
        ckp = torch.load(filename, map_location=device)
        hparams = ckp["hyper_parameters"]
        module = LeakersTrainingModule(**hparams)
        module.load_state_dict(ckp["state_dict"])
        return module.to(device)
