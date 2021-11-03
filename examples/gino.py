import torch
from leakers.nn.modules.base import BasicDecoder, BasicEncoder
from leakers.nn.modules.encoders.elastic import ElasticDecoder, ElasticEncoder
from leakers.nn.modules.shufflenet import ShuffleNet
import rich
import torchsummary

# model = ElasticEncoder(input_shape=[3, 64, 64]).to("cuda")
# model = BasicEncoder(input_size=[64, 64], input_channels=3).to("cuda")
# rich.print(
#     "[red] NUm Parameters: [/red]",
#     sum(p.numel() for p in model.parameters() if p.requires_grad),
# )

# # out = model(torch.rand(1, 3, 64, 64))
# torchsummary.summary(model, (3, 64, 64))


# model = ElasticDecoder(output_shape=[3, 64, 64], bn=True).to("cuda")
model = BasicDecoder(output_size=[64, 64], output_channels=3, latent_size=32).to("cuda")
rich.print(
    "[red] NUm Parameters: [/red]",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

# out = model(torch.rand(1, 3, 64, 64))
torchsummary.summary(model, (32,))
