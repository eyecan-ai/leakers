from torch.nn import modules
from leakers.nn.modules.base import LeakerModule
from leakers.nn.modules.factory import LeakerModuleFactory

import torch


def _plug_module_test(module: LeakerModule, image_shape, code_size):
    x = torch.randn(1, *image_shape)

    out = module.encode(x)
    assert isinstance(out, dict)
    assert "code" in out
    assert "rot_logits" in out
    assert "rot_classes" in out

    code = out["code"]
    assert code.shape == (1, code_size)

    img = module.generate(code)
    print("GENERATED", code.shape, img.shape)
    assert img.shape == (1, *image_shape)


class TestElasticCoder:
    def test_creation(self):

        for cin in [16, 32, 64]:
            for image_shape in [[3, 64, 64], [3, 32, 32]]:
                for code_size in [2, 4, 16, 32]:
                    for n_layers in [2, 3, 4]:
                        cfg = {
                            "type": "elastic",
                            "params": {
                                "image_shape": image_shape,
                                "code_size": code_size,
                                "cin": cin,
                                "n_layers": n_layers,
                                "k": 3,
                                "bn": False,
                                "act_middle": "torch.nn.ReLU",
                                "act_latent": "torch.nn.Tanh",
                                "act_last": "torch.nn.Sigmoid",
                            },
                        }

                        module = LeakerModuleFactory.create(cfg)
                        _plug_module_test(module, image_shape, code_size)
