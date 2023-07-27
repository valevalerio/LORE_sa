from .enc_dec import EncDec
import numpy as np
import copy

__all__ = ["EncDec", "OneHotEnc"]


class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    It relies on OneHotEncoder class from sklearn
    """

    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type='one-hot'
        self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)
        if self.dataset_descriptor.get("categoric") is None:
            raise Exception("Dataset descriptor is malformed for One-Hot Encoder: 'categoric' key is not present")

    def encode(self, x: np.array):
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)
        for k in self.encoded_descriptor['categoric'].keys():
            label_dict = self.encoded_descriptor['categoric'][k]
            label_index = label_dict['index']

            mapping = {}
            for value in range(len(label_dict['distinct_values'])):
                mapping[label_dict['distinct_values'][value]] = value

            arr = list(np.zeros(len(label_dict['distinct_values']), dtype=int))
            arr[mapping[x[label_index]]] = 1
            x = np.delete(x, label_index)
            x = np.insert(x, label_index, arr)

            #-1 is to keep the index starting from zero
            encoded_feature = {(label_index+i)-1:"=".join([k, v])  for i,v in enumerate(label_dict['distinct_values'])}
            self.encoded_features.update(encoded_feature)
            self.update_encoded_index(str(k),len(label_dict['distinct_values'])-1)
        return x

    def update_encoded_index(self,current_field, size: int):
        current_index_value = self.encoded_descriptor['categoric'][current_field]['index']
        for type in self.encoded_descriptor.keys():
            for k in self.encoded_descriptor[type]:
                if k != current_field:
                    original_index = self.encoded_descriptor[type][k]['index']
                    if original_index>current_index_value:
                        self.encoded_descriptor[type][k]['index'] = self.encoded_descriptor[type][k]['index'] + size

    def get_encoded_features(self):
        if self.encoded_features is None:
            raise Exception("You have not run the encoder yet")
        else:
            return self.encoded_features

    def __str__(self):
        if len(self.encoded_features) > 0:
            return "OneHotEncoder - features encoded: %s" % (",".join(self.encoded_features.keys()))
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
                mapping[list(mapping.keys())[l]] = [int(x) for x in arr]

            code = [int(x) for x in x[label_index: label_index + len(label_dict['distinct_values'])]]
            for t in mapping.keys():
                if list(mapping[t]) == code:
                    label = t

            x = np.concatenate((x[:label_index], [label], x[label_index+len(label_dict['distinct_values']):]),axis=0)

        return x
