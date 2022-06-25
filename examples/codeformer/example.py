import torch
from leakers.nn.modules.codeformer import EyePoolFormer, PoolFormerFactory
from leakers.nn.modules.codeformer import PoolFormerCoder

model = PoolFormerCoder(
    image_shape=(3, 32, 32),
    code_size=32,
    cin=32,
    n_layers=4,
    k=3,
    bn=True,
)
