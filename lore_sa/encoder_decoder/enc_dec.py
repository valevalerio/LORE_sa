from abc import abstractmethod
from lore_sa.dataset.dataset import Dataset

import pandas as pd

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self,):
        self.dataset = None
        self.class_name = None
        self.encdec = None
        self.features = None
        self.cate_features_names = None
        self.cate_features_idx = None
        self.type = None

    @abstractmethod
    def encode(self, x: Dataset, kwargs=None):
        return

    @abstractmethod
    def decode(self, x: Dataset, kwargs=None):
        return

    @abstractmethod
    def enc_fit_transform(self, dataset: Dataset =None, class_name: str =None):
        return



