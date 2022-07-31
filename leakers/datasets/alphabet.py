from abc import abstractmethod
import numpy as np
import math
from pyparsing import Optional
from scipy.spatial import KDTree
from pydantic import BaseModel, Extra, PrivateAttr


class AlphabetDataset:
    @classmethod
    def width_from_size(cls, size):
        return int(np.log2(size))

    @abstractmethod
    def words_to_indices(self, words: np.ndarray) -> np.ndarray:
        """Transform plain words [B,bit_size] into corresponding alphabet word number [B]

        :param words: input words [B, bit_size]
        :type words: np.ndarray
        :return: output alphabet numbers [B]
        :rtype: np.ndarray
        """
        pass


class BinaryAlphabetDataset(AlphabetDataset):
    def __init__(self, bit_size: int = 4, negative_range: bool = True):

        self._width = bit_size  # AlphabetDataset.width_from_size(self._size)
        self._size = 2**self._width
        self._negative_range = negative_range
        self._kdtree = None

    def _build_kdtree(self):
        data = []
        for sample in self:
            data.append(sample["x"])
        data = np.array(data)
        self._kdtree = KDTree(data=data)

    def _binary_repr(self, idx):
        binary_string = np.binary_repr(idx, width=self._width)
        bits = [float(x) for x in binary_string]
        if self._negative_range:
            bits = [x if x > 0 else -1.0 for x in bits]
        return bits

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        return {"x": np.array(self._binary_repr(idx)).astype(np.float32), "y": idx}

    def words_to_indices(self, words: np.ndarray) -> np.ndarray:

        if self._kdtree is None:
            self._build_kdtree()

        return self._kdtree.query(words)[1]


class DatasetBinaryAlphabet(AlphabetDataset, BaseModel, extra=Extra.forbid):

    width: int = 4
    negative_range: bool = True
    _kdtree = PrivateAttr()
    _size = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._size = 2**self.width
        self._kdtree = None

    def _build_kdtree(self):
        data = []
        for sample_id in range(len(self)):
            data.append(self[sample_id]["x"])
        data = np.array(data)
        self._kdtree = KDTree(data=data)

    def _binary_repr(self, idx):
        binary_string = np.binary_repr(idx, width=self.width)
        bits = [float(x) for x in binary_string]
        if self.negative_range:
            bits = [x if x > 0 else -1.0 for x in bits]
        return bits

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        return {"x": np.array(self._binary_repr(idx)).astype(np.float32), "y": idx}

    def words_to_indices(self, words: np.ndarray) -> np.ndarray:

        if self._kdtree is None:
            self._build_kdtree()

        return self._kdtree.query(words)[1]
