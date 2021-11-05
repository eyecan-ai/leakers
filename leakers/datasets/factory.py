from leakers.datasets.alphabet import BinaryAlphabetDataset


class AlphabetDatasetFactory:

    FACTORY_MAP = {"binary": BinaryAlphabetDataset}

    @classmethod
    def create(cls, cfg: dict):
        if cfg["type"] not in cls.FACTORY_MAP:
            raise ValueError(f"Unknown LeakerModule type: {cfg['type']}")

        return cls.FACTORY_MAP[cfg["type"]](**cfg["params"])
