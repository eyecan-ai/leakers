from pipelime.choixe.xconfig import XConfig
import rich
from leakers.nn.modules.elastic import ElasticCoder, CoderElastic
from leakers.datasets.alphabet import DatasetBinaryAlphabet
from leakers.nn.modules.randomizers import RandomizerVirtualCamera

cfg = XConfig.from_file("config.yml").process()
rich.print(cfg)

# dataset = cfg.dataset
# for idx in range(len(dataset)):
#     word = dataset[idx]
#     print("=" * 10)
#     print(word)
#     retrieved = dataset.words_to_indices(word["x"])
#     print(retrieved)


# print(dataset)
# coder = CoderElastic(shape=(3, 32, 32))
