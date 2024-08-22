from abc import ABC
from enum import Enum
from typing import Protocol

import shutil
import os
from num2words import num2words
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


class DatasetWrapper(ABC):
    def __init__(
        self, data_src_path: str, train: bool = True, download: bool = True
    ): ...

    @property
    def dataset(self): 
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    def label_to_str(self, i: int) -> str: ...

    def clean(self, path: str): ...


class MNISTWrapper(DatasetWrapper):
    def __init__(
        self, data_src_path: str, train: bool = True, download: bool = True
    ) -> DatasetWrapper:
        self.dataset = MNIST(data_src_path, train=train, download=download)
        self.label_to_str = num2words
        self.num_classes = len(self.dataset.classes)
        self.clean = lambda path: shutil.rmtree(
            os.path.join(path, "MNIST"), ignore_errors=True
        )


class CIFAR10Wrapper(DatasetWrapper):
    def __init__(
        self, data_src_path: str, train: bool = True, download: bool = True
    ) -> DatasetWrapper:
        self.dataset = CIFAR10(data_src_path, train=train, download=download)
        self.label_to_str = lambda i: self.dataset.classes[i]
        self.num_classes = len(self.dataset.classes)

    def clean(self, data_path: str):
        os.remove(os.path.join(data_path, "cifar-10-python.tar.gz"))
        shutil.rmtree(
            os.path.join(data_path, "cifar-10-batches-py"),
            ignore_errors=True,
        )


class CIFAR100Wrapper(DatasetWrapper):
    def __init__(
        self, data_src_path: str, train: bool = True, download: bool = True
    ) -> DatasetWrapper:
        self.dataset = CIFAR100(data_src_path, train=train, download=download)
        self.label_to_str = lambda i: self.dataset.classes[i]
        self.num_classes = len(self.dataset.classes)

    def clean(self, data_path: str):
        os.remove(os.path.join(data_path, "cifar-100-python.tar.gz"))
        shutil.rmtree(
            os.path.join(data_path, "cifar-100-python"),
            ignore_errors=True,
        )


class TextureType(Enum):
    APPLES = "apples"
    OBSTACLES = "obstacles"
    GABORS = "gabors"
    MNIST = "mnist"
    CIFAR10 = "cifar-10"
    CIFAR100 = "cifar-100"

    @property
    def is_asset(self):
        return self in [TextureType.APPLES, TextureType.OBSTACLES, TextureType.GABORS]

    @property
    def is_dataset(self):
        return not self.is_asset

    def out_dir(self, test: bool) -> str:
        _dir = self.value
        if test:
            assert self.is_dataset, "Test set is only possible for Datasets"
            _dir += "-test"
        return _dir

    def get_dataset_wrapper(
        self, data_src_path: str, train: bool = True, download: bool = True
    ) -> DatasetWrapper:
        assert self.is_dataset, "Only datasets should  a DatasetWrapper!"
        if self is TextureType.MNIST:
            wrapper = MNISTWrapper(data_src_path, train, download)
        elif self is TextureType.CIFAR10:
            wrapper = MNISTWrapper(data_src_path, train, download)
        elif self is TextureType.CIFAR100:
            wrapper = MNISTWrapper(data_src_path, train, download)
        return wrapper
