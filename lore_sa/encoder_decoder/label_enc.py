import copy

from .enc_dec import EncDec
import numpy as np


__all__ = ["EncDec", "LabelEnc"]
class LabelEnc(EncDec):
    """
    It provides an interface to access Label enconding functions.
    """

    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type = "label"
        if self.dataset_descriptor.get("ordinal") is None:
            raise Exception("Dataset descriptor is malformed for Label Encoder: 'ordinal' key is not present")

    def encode(self, x: np.array):
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)
        for k in self.dataset_descriptor["ordinal"].keys():
            label_index = self.dataset_descriptor["ordinal"][k]['index']
            values_dict = {k: v for v, k in enumerate(self.dataset_descriptor["ordinal"][k]['distinct_values'])}
            x[label_index] = values_dict[x[label_index]]
            self.encoded_features.update({k:label_index})
        return x

    def get_encoded_features(self):
        if self.encoded_features is None:
            raise Exception("You have not run the encoder yet")
        else:
            return self.encoded_features

    def __str__(self):
        if len(self.encoded_features) > 0:
            return "LabelEncoder - features encoded: %s" % (",".join(self.encoded_features.keys()))
        else:
            return "LabelEncoder - no features encoded"

    def decode(self, x: np.array):
        """
        Decode the array staring from the original descriptor

        :param [Numpy array] x: Array to decode
        :return [Numpy array]: Decoded array
        """
        for k in self.dataset_descriptor["ordinal"].keys():
            label_index = self.dataset_descriptor["ordinal"][k]['index']
            values_dict = {v: k for v, k in enumerate(self.dataset_descriptor["ordinal"][k]['distinct_values'])}
            x[label_index] = values_dict[int(x[label_index])]
        return x