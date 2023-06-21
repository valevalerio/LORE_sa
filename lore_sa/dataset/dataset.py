
import pandas as pd
from pandas import DataFrame

__all__ = ["Dataset"]

class Dataset():
    """
    It provides an interface to handle datasets, including some essential information on the structure and
    semantic of the dataset.
    """
    def __init__(self,data: DataFrame, class_name:str = None):
        self.encdec = None
        self.class_name = class_name
        self.df = data
        self.feature_names = None
        self.class_values = None
        self.numeric_columns = None
        self.real_feature_names = None
        self.features_map = None
        self.rdf = data
        

    @classmethod
    def from_csv(cls, filename: str, class_name: str=None):
        """
        Read a comma-separated values (csv) file into Dataset object.
        :param [str] filename:
        :param class_name: optional
        :return:
        """
        df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
        dataset_obj = cls(df, class_name=class_name)
        dataset_obj.filename = filename
        return dataset_obj

    @classmethod
    def from_dict(cls, data: dict, class_name: str=None):
        """
        From dicts of Series, arrays, or dicts.
        :param [dict] data:
        :param class_name: optional
        :return:
        """
        return cls(pd.DataFrame(data), class_name=class_name)

    def set_class_name(self,class_name: str):
        """
        Set the class name. Only the column name string
        :param [str] class_name:
        :return:
        """
        self.class_name = class_name

    def get_class_values(self):
        """
        Provides the class_name
        :return:
        """
        if self.class_name is None:
            raise Exception("ERR: class_name is None. Set class_name with set_class_name('<column name>')")
        return self.df[self.class_name].values


    def get_numeric_columns(self):
        numeric_columns = list(self.df._get_numeric_data().columns)
        return numeric_columns

    def get_original_dataset(self):
        """
        Provides the original dataset
        :return:
        """
        return self.rdf

    def create_feature_map(self):
        """
        Provides a dictionary with the list of numeric and categorical columns.
        :return:
        """
        self.features_map = dict()
        self.features_map["numeric_columns"] = self.get_numeric_columns()
        self.features_map["categorical_columns"] = [c for c in self.df.columns if c not in self.get_numeric_columns()]
        return self.features_map
