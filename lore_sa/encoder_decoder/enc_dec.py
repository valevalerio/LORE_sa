from abc import abstractmethod

import pandas as pd

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self, dataset=None, class_name = None):
        self.dataset = dataset
        self.class_name = class_name
        self.encdec = None
        self.features = None
        self.cate_features_names = None
        self.cate_features_idx = None
        self.type = None

    @abstractmethod
    def enc(self, x: list, y: list, kwargs=None):
        return

    @abstractmethod
    def dec(self, x: list, kwargs=None):
        return

    @abstractmethod
    def enc_fit_transform(self, dataset=None, class_name=None):
        return

    @abstractmethod
    def prepare_dataset(self, df: pd.DataFrame, class_name, numeric_columns, rdf):
        """
        This function prepare the dataset in a way to use the encoder with the function enc.

        :param [Dataframe] df: the original dataframe
        :param [str] class_name: class name target
        :param [int] numeric_columns: list of numeric columns indexes
        :param [Dataframe] rdf: original dataframe ?
        :return [Dataframe] df_dummy:  encoded dataframe
        :return feature_names:
        :return features_map:
        :return numeric_columns:
        :return class_values:
        :return rdf:
        :return real_feature_names:
        """
        return


