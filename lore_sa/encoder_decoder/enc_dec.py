from abc import abstractmethod
import numpy as np

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self,dataset_descriptor):
        self.dataset_descriptor = dataset_descriptor
        self.encoded_features = []

    @abstractmethod
    def encode(self, x: np.array):
        """
        It applies the encoder to the input features

        :param[Numpy array] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        return

    @abstractmethod
    def decode(self, x: np.array):
        return
