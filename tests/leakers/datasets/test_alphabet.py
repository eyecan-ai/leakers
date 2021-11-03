from leakers.datasets.alphabet import BinaryAlphabetDataset
import numpy as np


class TestBinaryAlphabetDataset:
    def test_dataset(self):

        for bs in range(0, 16):

            dataset = BinaryAlphabetDataset(bit_size=bs, negative_range=False)

            assert bs == dataset.width_from_size(len(dataset))

            words = []
            for idx, sample in enumerate(dataset):
                assert idx == sample["y"]
                words.append(sample["x"])

            assert len(dataset) == len(np.unique(words, axis=1))

            indices = dataset.words_to_indices(words)
            assert np.array_equal(indices, np.array(range(len(dataset))))
