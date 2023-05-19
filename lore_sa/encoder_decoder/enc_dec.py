from abc import abstractmethod

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self, dataset=None, class_name = None):
        self.dataset = dataset
        self.class_name = class_name
        self.encdec = None
        self.features = None
        self.cate_features_names = None
        self.cate_features_idx = None
        self.type = None

    @abstractmethod
    def enc(self, x: list, y: list, kwargs=None):
        return

    @abstractmethod
    def dec(self, x: list, kwargs=None):
        return

    @abstractmethod
    def enc_fit_transform(self, dataset=None, class_name=None):
        return



