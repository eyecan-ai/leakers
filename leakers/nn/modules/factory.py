from leakers.nn.modules.codeformer import PoolFormerCoder
from leakers.nn.modules.elastic import ElasticCoder
from leakers.nn.modules.randomizers import VirtualRandomizer


class LeakerModuleFactory:

    FACTORY_MAP = {
        "elastic": ElasticCoder,
        "codeformer": PoolFormerCoder,
    }

    @classmethod
    def create(cls, cfg: dict):
        if cfg["type"] not in cls.FACTORY_MAP:
            raise ValueError(f"Unknown LeakerModule type: {cfg['type']}")

        return cls.FACTORY_MAP[cfg["type"]](**cfg["params"])


class RandomizersFactory:

    FACTORY_MAP = {"virtual": VirtualRandomizer}

    @classmethod
    def create(cls, cfg: dict):
        if cfg["type"] not in cls.FACTORY_MAP:
            raise ValueError(f"Unknown Randomizer type: {cfg['type']}")

        return cls.FACTORY_MAP[cfg["type"]](**cfg["params"])
