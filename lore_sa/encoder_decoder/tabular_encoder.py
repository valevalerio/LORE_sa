from lore_sa.encoder_decoder import EncDec, LabelEnc, OneHotEnc
from lore_sa.logger import logger
import numpy as np

__all__ = ["EncDec", "TabularEnc","LabelEnc"]
class TabularEnc(EncDec):
    """
    It combine different encoders/decoders over a table dataset
    """
    def __init__(self,descriptor: dict, target_class: str = None):
        super().__init__(descriptor)
        self.target_class = target_class
        self.type='tabular'

    def encode(self, x: np.array):
        """
        Combine label encoding, as first, and one-hot encoding to provide a table encoded.
        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        self.one_hot_enc = OneHotEnc(self.dataset_descriptor)
        one_hot_encoded = self.one_hot_enc.encode(x)
        self.encoded_features.update(self.one_hot_enc.get_encoded_features())

        new_descriptor = self.set_target_label(self.one_hot_enc.encoded_descriptor)

        self.label_enc = LabelEnc(new_descriptor)
        label_encoded = self.label_enc.encode(one_hot_encoded)
        self.encoded_features.update(self.label_enc.get_encoded_features())
        return np.array([int(n) for n in label_encoded])

    def set_target_label(self, descriptor):
        if self.target_class is None:
            logger.warning("No target class is defined")
            return descriptor

        for type in descriptor:
            for k in descriptor[type]:
                if k == self.target_class:
                    descriptor['target'] = k
                    descriptor['type'].pop(k)
                    return descriptor
                else:
                    logger.warning("No target class is finded")
                    return descriptor


    def get_encoded_features(self):
        if self.encoded_features is None:
            raise Exception("You have not run the encoder yet")
        else:
            return self.encoded_features

    def decode(self, x: np.array):
        """
        Combine one-hot decoding, as first, and label encoding to provide a table encoded.
        :param x:
        :return:
        """
        label_enc = LabelEnc(self.dataset_descriptor)
        one_hot_enc = OneHotEnc(self.dataset_descriptor)
        one_hot_decoded = one_hot_enc.decode(x)
        label_decoded = label_enc.decode(one_hot_decoded)
        return label_decoded
