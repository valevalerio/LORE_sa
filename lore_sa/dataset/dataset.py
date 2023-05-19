import numpy as np
import pandas as pd

from collections import defaultdict
from lore_sa.encoder_decoder import EncDec

__all__ = ["DataSet"]

class DataSet():
    """
    This class provides an interface to handle dataset such as tabular, images etc...
    Dataset class incapsulates the data and expose methods to prepare the dataset.
    """
    def __init__(self,filename: str, class_name: list):
        self.original_filename = filename
        self.class_name = class_name
        self.df = None
        self.feature_names = None
        self.class_values = None
        self.numeric_columns = None
        self.real_feature_names = None
        self.features_map = None
        self.rdf = None

    def prepare_dataset(self, encdec: EncDec = None):
        """
        The method prepare_dataframe scans the dataset and extract the following information

        :param filename:
        :param [str] class_name:
        :return:
            -   df  - is a trasformed version of the original dataframe, where discrete attributes are transformed into numerical attributes by using one hot encoding strategy;
            -   feature_names - is a list containint the names of the features after the transformation;
            -   class_values - the list of all the possible values for the class_field column;
            -   numeric_columns - a list of the original features that contain numeric (i.e. continuous) values;
            -   rdf -  the original dataframe, before the transformation;
            -   real_feature_names - the list of the features of the dataframe before the transformation;
            -   features_map - it is a dictionary pointing each feature to the original one before the transformation.
        """

        df_original = pd.read_csv(self.original_filename, skipinitialspace=True, na_values='?', keep_default_na=True)

        df = self.__remove_missing_values(df_original)
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

        return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map

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

    def get_k(self):
        return self.df

    def get_feature_map(self):
        return self.feature_map

    def get_class_values(self):
        return self.class_values

    def get_numeric_columns(self):
        return self.numeric_columns

    def get_original_dataset(self):
        return self.rdf

