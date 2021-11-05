from leakers.trainers.modules import LeakersTrainingModule


class TestLeakersTrainingModule:
    def test_creation(self):

        code_size = 8
        hparams = {
            "coder": {
                "type": "elastic",
                "params": {
                    "image_shape": [3, 64, 64],
                    "code_size": 8,
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
                    "color_jitter": False,
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

        module = LeakersTrainingModule(**hparams)

        print(module.proto_dataset)
