from .enc_dec import EncDec
import pandas as pd
import numpy as np
from lore_sa.dataset.tabular_dataset import TabularDataset

__all__ = ["EncDec", "OneHotEnc"]


class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    It relies on OneHotEncoder class from sklearn
    """

    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type='one-hot'

    def encode(self, x: np.array):
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """

        for k in self.dataset_descriptor['categoric'].keys():
            label_dict = self.dataset_descriptor['categoric'][k]
            label_index = label_dict['index']

            mapping = {}
            for value in range(len(label_dict['distinct_values'])):
                mapping[label_dict['distinct_values'][value]] = value

            arr = list(np.zeros(len(label_dict['distinct_values']), dtype=int))
            arr[mapping[x[label_index]]] = 1
            x = np.delete(x, label_index)
            x = np.insert(x, label_index, arr)

            self.encoded_features.append(k)
            self.update_encoded_index(k,len(label_dict['distinct_values'])-1)
        return x

    def update_encoded_index(self,current_field, size: int):
        current_index_value = self.dataset_descriptor['categoric'][current_field]['index']
        for type in self.dataset_descriptor.keys():
            for k in self.dataset_descriptor[type]:
                if k != current_field:
                    original_index = self.dataset_descriptor[type][k]['index']
                    if original_index>current_index_value:
                        self.dataset_descriptor[type][k]['index'] = self.dataset_descriptor[type][k]['index'] + size

    def __str__(self):
        if len(self.encoded_features) > 0:
            return "OneHotEncoder - features encoded: %s" % (",".join(self.encoded_features))
        else:
            return "OneHotEncoder - no features encoded"

    def decode(self, x: np.array):
        """
        Decode the array staring from the original descriptor

        :param [Numpy array] x: Array to decode
        :return [Numpy array]: Decoded array
        """
        for k in self.dataset_descriptor['categoric'].keys():
            label_dict = self.dataset_descriptor['categoric'][k]
            label_index = label_dict['index']

            mapping = {}
            for value in range(len(label_dict['distinct_values'])):
                mapping[label_dict['distinct_values'][value]] = value


            for l in range(len(label_dict['distinct_values'])):
                arr = list(np.zeros(len(label_dict['distinct_values']), dtype=int))
                arr[l] = 1
                mapping[list(mapping.keys())[l]] = [str(x) for x in arr]

            for t in mapping.keys():
                if list(mapping[t]) == list(x[label_index: label_index + len(label_dict['distinct_values'])]):
                    label = t

            x = np.concatenate((x[:label_index], [label], x[label_index+len(label_dict['distinct_values']):]),axis=0)

        return x
