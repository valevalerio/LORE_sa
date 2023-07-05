from lore_sa.encoder_decoder import EncDec, LabelEnc, OneHotEnc
import numpy as np

__all__ = ["EncDec", "TabularEnc","LabelEnc"]
class TabularEnc(EncDec):
    """
    It combine different encoders/decoders over a table dataset
    """
    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type='tabular'

    def encode(self, x: np.array):
        """
        Combine label encoding, as first, and one-hot encoding to provide a table encoded.
        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        label_enc = LabelEnc(self.dataset_descriptor)
        one_hot_enc = OneHotEnc(self.dataset_descriptor)
        label_encoded = label_enc.encode(x)
        one_hot_encoded = one_hot_enc.encode(label_encoded)
        return np.array([int(n) for n in one_hot_encoded])

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