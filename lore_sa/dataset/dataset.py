import numpy as np
import pandas as pd
from pandas import DataFrame

from collections import defaultdict
from lore_sa.encoder_decoder import EncDec

__all__ = ["DataSet"]

class DataSet():
    """
    This class provides an interface to handle dataset such as tabular, images etc...
    Dataset class incapsulates the data and expose methods to prepare the dataset.
    """
    def __init__(self,data: DataFrame):
        self.class_name = None
        self.df = data
        self.feature_names = None
        self.class_values = None
        self.numeric_columns = None
        self.real_feature_names = None
        self.features_map = None
        self.rdf = None
        self.filename = None

    @classmethod
    def from_csv(cls, filename: str):
        """
        Read a comma-separated values (csv) file into Dataset object.
        :param [str] filename:
        :return:
        """
        df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
        dataset_obj = cls(df)
        dataset_obj.filename = filename
        return dataset_obj

    @classmethod
    def from_dict(cls, data: dict):
        """
        From dicts of Series, arrays, or dicts.
        :param [dict] data:
        :return:
        """
        return cls(pd.DataFrame(data))

    def set_class_name(self,class_name: str):
        self.class_name = class_name

    def prepare_dataset(self, encdec: EncDec = None):
        """
        The method prepare_dataframe scans the dataset and extract the following information

        :param [EncDec] encdec: Encoder Object
        """

        df = self.__remove_missing_values(self.df)
        rdf = df
        self.df = df
        self.rdf = rdf
        numeric_columns = self.__get_numeric_columns(self.df)

        if self.class_name in numeric_columns:
            numeric_columns.remove(self.class_name)

        if encdec.type == 'onehot' or encdec is None:
            df, feature_names, class_values = self.__one_hot_encoding(self.df, self.class_name)
            real_feature_names = self.__get_real_feature_names(self.rdf, numeric_columns, self.class_name)
            self.rdf = self.rdf[real_feature_names + (class_values if isinstance(self.class_name, list) else [self.class_name])]
            features_map = self.__get_features_map(feature_names, real_feature_names)

        elif encdec.type == 'target':
            feature_names = self.df.columns.values
            feature_names = np.delete(feature_names, np.where(feature_names == self.class_name))
            class_values = np.unique(self.df[self.class_name]).tolist()
            numeric_columns = list(self.df._get_numeric_data().columns)
            real_feature_names = [c for c in self.df.columns if c in numeric_columns and c != self.class_name]
            real_feature_names += [c for c in self.df.columns if c not in numeric_columns and c != self.class_name]
            features_map = dict()
            for f in range(0, len(real_feature_names)):
                features_map[f] = dict()
                features_map[f][real_feature_names[f]] = np.where(feature_names == real_feature_names[f])[0][0]


        self.feature_names = feature_names
        self.class_values = class_values
        self.numeric_columns = numeric_columns
        self.real_feature_names = real_feature_names
        self.features_map = features_map

    def __remove_missing_values(self,df):
        for column_name, nbr_missing in df.isna().sum().to_dict().items():
            if nbr_missing > 0:
                if column_name in df._get_numeric_data().columns:
                    mean = df[column_name].mean()
                    df[column_name].fillna(mean, inplace=True)
                else:
                    mode = df[column_name].mode().values[0]
                    df[column_name].fillna(mode, inplace=True)
        return df

    def __get_numeric_columns(self,df):
        numeric_columns = list(df._get_numeric_data().columns)
        return numeric_columns

    def __get_features_map(slef,feature_names, real_feature_names):
        features_map = defaultdict(dict)
        i = 0
        j = 0

        while i < len(feature_names) and j < len(real_feature_names):
            if feature_names[i] == real_feature_names[j]:
                #print('in if ', feature_names[i], real_feature_names[j])
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
                j += 1

            elif feature_names[i].startswith(real_feature_names[j]):
                #print('in elif ', feature_names[i], real_feature_names[j])
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
            else:
                j += 1
        return features_map


    def __get_real_feature_names(self,rdf, numeric_columns, class_name):
        if isinstance(class_name, list):
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
        else:
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
        return real_feature_names


    def __one_hot_encoding(self, df, class_name):
        if not isinstance(class_name, list):
            dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
            class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
            dfY = df[class_name].map(class_name_map)
            df = pd.concat([dfX, dfY], axis=1)
            df =df.reindex(dfX.index)
            feature_names = list(dfX.columns)
            class_values = sorted(class_name_map)
        else:
            dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
            # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
            class_values = sorted(class_name)
            dfY = df[class_values]
            df = pd.concat([dfX, dfY], axis=1)
            df = df.reindex(dfX.index)
            feature_names = list(dfX.columns)
        return df, feature_names, class_values

    def get_feature_map(self):
        return self.feature_map

    def get_class_values(self):
        return self.class_values

    def get_numeric_columns(self):
        return self.numeric_columns

    def get_original_dataset(self):
        return self.rdf

