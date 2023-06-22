from .enc_dec import EncDec
import pandas as pd

from lore_sa.dataset.dataset import Dataset
from sklearn.preprocessing import LabelEncoder

__all__ = ["EncDec", "LabelEnc"]
class LabelEnc(EncDec):
    """
    It provides an interface to access Label enconding functions.
    """

    def __init__(self):
        super().__init__()
        self.type = "label"
        self.feature_encoding = {}

    def encode(self, dataset: Dataset, features_to_encode: list):
        """
        It applies the encoder to the input features

        :param [Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        self.dataset_encoded = pd.DataFrame()
        for feature in features_to_encode:
            self.dataset_encoded[feature] = labelencoder.fit_transform(dataset.df[feature])
            self.feature_encoding[feature] = {i:v for i,v in enumerate(labelencoder.classes_)}


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

    def decode(self, dataset: Dataset, feature_encoding: dict = None):
        """
        Provides a new dataframe decoded from dictionary of encoding features

        :param [Dataset] dataset: Dataset to decode
        :param feature_encoding: a dictionary to convert the values from numeric to string.
        :return:
        """
        self.dataset_decoded = pd.DataFrame()
        if feature_encoding is None:
            feature_encoding = self.feature_encoding
        for feature in feature_encoding:
            self.dataset_decoded[feature] = dataset.df[feature].apply(lambda x: feature_encoding[feature][x])

        dataset.df = pd.concat([dataset.df.drop(columns=feature_encoding.keys()), self.dataset_decoded], axis=1)
        return dataset.df