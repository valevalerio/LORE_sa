
import pandas as pd
from pandas import DataFrame

from lore_sa.encoder_decoder import EncDec

__all__ = ["DataSet"]

class DataSet():
    """
    It provides an interface to handle datasets, including some essential information on the structure and
    semantic of the dataset.
    """
    def __init__(self,data: DataFrame):
        self.encdec = None
        self.class_name = None
        self.df = data
        self.feature_names = None
        self.class_values = None
        self.numeric_columns = None
        self.real_feature_names = None
        self.features_map = None
        self.rdf = None
        

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

    # def prepare_dataset(self, encdec: EncDec = None):
    #     """
    #     It tranforms the dataset in order to be ready for neighborhood generation, eventually applying 
    #     the input encoder

    #     :param [EncDec] encdec: Encoder Object
    #     """
    #     self.encdec = encdec
    #     df = self.__remove_missing_values(self.df)
    #     rdf = df
    #     self.df = df
    #     self.rdf = rdf
    #     self.numeric_columns = self.__get_numeric_columns(self.df)

    #     if self.class_name is None:
    #         raise ValueError("class_name is None. Set it with objects.set_class_name('target')")

    #     if self.class_name in self.numeric_columns:
    #         self.numeric_columns.remove(self.class_name)

    #     self.df, self.feature_names, self.features_map, self.numeric_columns, self.class_values, self.rdf, self.real_feature_names = self.encdec.prepare_dataset(self.df, self.class_name, self.numeric_columns, self.rdf)

    # def __remove_missing_values(self,df):
    #     for column_name, nbr_missing in df.isna().sum().to_dict().items():
    #         if nbr_missing > 0:
    #             if column_name in df._get_numeric_data().columns:
    #                 mean = df[column_name].mean()
    #                 df[column_name].fillna(mean, inplace=True)
    #             else:
    #                 mode = df[column_name].mode().values[0]
    #                 df[column_name].fillna(mode, inplace=True)
    #     return df

    # def __get_numeric_columns(self,df):
    #     numeric_columns = list(df._get_numeric_data().columns)
    #     return numeric_columns

    # def get_feature_map(self):
    #     """
    #     feature_map is a dictionary to track one-hot-encoded features, implemented as dataframe columns. An example of the format of suct a dictionary is: 
            # {
            #   0: {'age': 0},
            #   2: {'sex = Male': 2,
            #       'sex = Female': 3,
            #   }
            # ...},
    #     Here, the key of the dictionary is the column index of the feature, while the value is a further dictionary cointaining
    #     single column name or eventually features grouped after the trasformation of a feature through one hot encoding
    #     """
    #     return self.feature_map

    def get_class_values(self):
        return self.class_values

    def get_numeric_columns(self):
        numeric_columns = list(self.df._get_numeric_data().columns)
        return numeric_columns

    def get_original_dataset(self):
        return self.rdf

