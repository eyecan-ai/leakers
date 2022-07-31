from leakers.datasets.datamodules import GenericAlphabetDatamodule
from leakers.trainers.modules import TrainableLeakers
from leakers.datasets.factory import AlphabetDatasetFactory
import pytorch_lightning as pl


class TestLeakersTrainingModule:
    def test_creation(self):

        code_size = 4
        hparams = {
            "coder": {
                "type": "elastic",
                "params": {
                    "image_shape": [3, 64, 64],
                    "code_size": code_size,
                    "cin": 32,
                    "n_layers": 4,
                    "k": 3,
                    "bn": True,
                    "act_middle": "torch.nn.ReLU",
                    "act_latent": None,
                    "act_last": "torch.nn.Sigmoid",
                },
            },
            "randomizer": {
                "type": "virtual",
                "params": {
                    "warping": True,
                    "color_jitter": True,
                    "random_erasing": True,
                    "random_erasing_p": 0.1,
                },
            },
            "dataset": {"type": "binary", "params": {"bit_size": code_size}},
            "losses": {
                "code_loss": "torch.nn.SmoothL1Loss",
                "code_loss_weight": 1.0,
                "rot_loss_weight": 0.1,
            },
            "rotations": {"randomize": False},
            "optimizer": {
                "lr": 0.0001,
            },
        }

        module = TrainableLeakers(**hparams)
        dataset = AlphabetDatasetFactory.create(hparams["dataset"])

        datamodule = GenericAlphabetDatamodule(
            dataset=dataset, batch_size=2**code_size
        )

        trainer = pl.Trainer(
            gpus=0,
            max_epochs=1,
            log_every_n_steps=10,
            default_root_dir="/tmp/",
            check_val_every_n_epoch=100,
        )

        trainer.fit(module, datamodule)
