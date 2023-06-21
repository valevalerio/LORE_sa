from .enc_dec import EncDec
import pandas as pd

from lore_sa.dataset.dataset import Dataset
from sklearn.preprocessing import LabelEncoder

__all__ = ["EncDec", "LabelEnc"]
class LabelEnc(EncDec):
    """
    It provides an interface to access Label enconding (https://en.wikipedia.org/wiki/One-hot) functions.
    """

    def __init__(self):
        super().__init__()
        self.type = "label"

    def encode(self, dataset: Dataset, features_to_encode: list):
        """
        It applies the encoder to the input features

        :param [Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        # creating instance of labelencoder
        labelencoder = LabelEncoder()

        self.dataset_encoded = dataset.df[features_to_encode].apply(labelencoder.fit_transform)

        # buffering original features to get them eventually decoded
        self.original_data = dataset.df[features_to_encode]
        self.original_features = features_to_encode
        self.encoded_features = self.dataset_encoded.columns

        # substituing old features with encoded
        dataset.df = pd.concat([dataset.df.drop(columns=features_to_encode), self.dataset_encoded], axis=1)

        return dataset.df

    def __str__(self):
        if self.encoded_features is not None:
            return "LabelEncoder - features encoded: %s" % (",".join(self.original_features))
        else:
            return "LabelEncoder - no features encoded"