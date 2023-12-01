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
        self.encoded_features = {}
        self.encoded_descriptor = None

    @abstractmethod
    def encode(self, x: np.array) -> np.array:
        """
        It applies the encoder to the input features

        :param[Numpy array] x: the Dataset containing the features to be encoded
        """
        return

    @abstractmethod
    def get_encoded_features(self):
        """
        Provides a dictionary with the new encoded features name and the new index
        :return:
        """
        return

    @abstractmethod
    def decode(self, x: np.array) -> np.array:
        return


    @abstractmethod
    def decode_target_class(self, x: np.array):
        return


class IdentityEncoder(EncDec):
    """
    It provides an interface to access Identity encoding functions.
    """
    def __init__(self,dataset_descriptor):
        super().__init__(dataset_descriptor)

    def encode(self, x: np.array) -> np.array:
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        return x

    def get_encoded_features(self):
        """
        Provides a dictionary with the new encoded features name and the new index
        :return:
        """
        return self.encoded_features

    def decode(self, x: np.array) -> np.array:
        return x

    def decode_target_class(self, x: np.array):
        return x