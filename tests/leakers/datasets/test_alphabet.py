from leakers.datasets.alphabet import (
    AlphabetDataset,
    BinaryAlphabetDataset,
)
import numpy as np

from leakers.datasets.factory import AlphabetDatasetFactory


class TestBinaryAlphabetDataset:
    def _dataset_plug_test(self, dataset: AlphabetDataset, bs: int):
        assert bs == dataset.width_from_size(len(dataset))

        words = []
        for idx, sample in enumerate(dataset):
            assert idx == sample["y"]
            words.append(sample["x"])

        assert len(dataset) == len(np.unique(words, axis=1))

        indices = dataset.words_to_indices(words)
        assert np.array_equal(indices, np.array(range(len(dataset))))

    def test_dataset(self):

        for bs in range(0, 16, 2):
            for negative_range in [True, False]:

                dataset = BinaryAlphabetDataset(
                    bit_size=bs, negative_range=negative_range
                )

                self._dataset_plug_test(dataset, bs)

    def test_creation(self):

        for bs in range(0, 16, 2):
            for negative_range in [True, False]:
                cfg = {
                    "type": "binary",
                    "params": {"bit_size": bs, "negative_range": negative_range},
                }

                dataset = AlphabetDatasetFactory.create(cfg)
                self._dataset_plug_test(dataset, bs)
