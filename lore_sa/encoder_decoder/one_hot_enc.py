from collections import defaultdict

from .enc_dec import EncDec
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from lore_sa.dataset.dataset import Dataset

__all__ = ["EncDec","OneHotEnc"]

extend: OneHotEncoder
class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    """
    def __init__(self):
        self.dataset_enc = None
        self.original_features = None
        self.original_data = None
        self.encoded_features = None


    def encode(self, dataset: Dataset, features_to_encode):
        """
        It applies the encoder to the input features

        :param[Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        encoded = pd.get_dummies(dataset.df[features_to_encode], prefix_sep='=')
        
        #buffering original features to get them eventually decoded
        self.original_data = dataset.df[features_to_encode]
        self.original_features = features_to_encode
        self.encoded_features = encoded.columns

        #substituing old features with encoded 
        

        dataset.df = pd.concat([dataset.df.drop(columns = features_to_encode),encoded ], axis = 1)
        
        return dataset.df

    def __str__(self):
        if self.encoded_features is not None: 

            return ("OneHotEncoder - features encoded: %s"%(",".join(self.original_features)))
        else:
            return ("OneHotEncoder - no features encoded")
    

    def decode(self, dataset: Dataset, kwargs=None):
        if self.encoded_features is not None:
            return pd.concat([dataset.df.drop(columns =self.encoded_features), self.original_data] , axis =1)
        else:
            return dataset.df

       