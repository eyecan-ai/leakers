import torch
import rich
import torchsummary

from leakers.nn.modules.elastic import ElasticCoder, ElasticDecoder, ElasticEncoder


cin = 32
code_size = 32
model = ElasticCoder(image_shape=[3, 64, 64], cin=cin, code_size=code_size).to("cuda")
# model = BasicEncoder(input_size=[64, 64], input_channels=3).to("cuda")
rich.print(
    "[red] NUm Parameters: [/red]",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

# out = model(torch.rand(1, 3, 64, 64))
torchsummary.summary(model, (code_size,))


# model = ElasticDecoder(output_shape=[3, 64, 64], bn=True).to("cuda")
# model = BasicDecoder(output_size=[64, 64], output_channels=3, latent_size=32).to("cuda")
# rich.print(
#     "[red] NUm Parameters: [/red]",
#     sum(p.numel() for p in model.parameters() if p.requires_grad),
# )

# # out = model(torch.rand(1, 3, 64, 64))
# torchsummary.summary(model, (32,))
