import torch


import rich

from leakers.trainers.modules import LeakersTrainingModule, RuneTrainingModule


class LeakersInferenceModuleFactory:
    @classmethod
    def create_from_checkpoint(cls, filename: str, device="cpu"):
        ckp = torch.load(filename, map_location=device)
        hparams = ckp["hyper_parameters"]
        module = LeakersTrainingModule(**hparams)
        module.load_state_dict(ckp["state_dict"])
        return module.to(device)


class LeakersConfigurationsBucket:

    CONFIGURATIONS_MAP = {
        "default": {
            "coder": {
                "type": "elastic",
                "params": {
                    "image_shape": [
                        "@int(channels)",
                        "@int(image_size)",
                        "@int(image_size)",
                    ],
                    "code_size": "@int(code_size)",
                    "cin": 32,
                    "n_layers": 4,
                    "k": 3,
                    "bn": False,
                    "act_middle": "torch.nn.ReLU",
                    "act_latent": None,
                    "act_last": "torch.nn.Sigmoid",
                },
            },
            "randomizer": {
                "type": "virtual",
                "params": {
                    "image_shape": [
                        "@int(channels)",
                        "@int(image_size)",
                        "@int(image_size)",
                    ],
                    "color_jitter": True,
                    "random_erasing": True,
                    "warper": True,
                },
            },
            "dataset": {"type": "binary", "params": {"bit_size": "@int(code_size)"}},
            "losses": {
                "code_loss": "torch.nn.SmoothL1Loss",
                "code_loss_weight": 1.0,
                "rot_loss_weight": 0.1,
            },
            "rotations": {"randomize": False},
            "optimizer": {
                "lr": 0.0001,
            },
            "training": {"randomizer_warmup_epochs": 2},
        }
    }


class RunesInferenceModuleFactory:
    @classmethod
    def create_from_checkpoint(cls, filename: str, device="cpu"):
        ckp = torch.load(filename, map_location=device)
        hparams = ckp["hyper_parameters"]
        module = RuneTrainingModule(**hparams)
        module.load_state_dict(ckp["state_dict"])
        return module.to(device)
