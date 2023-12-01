from lore_sa.dataset import Dataset

from lore_sa.logger import logger
import pandas as pd
from pandas import DataFrame
import numpy as np

__all__ = ["TabularDataset","Dataset"]


class TabularDataset(Dataset):
    """
    It provides an interface to handle datasets, including some essential information on the structure and
    semantic of the dataset.

    Attributes:
        df (pandas.DataFrame): dataframe containing the whole dataset
        descriptor (dict): it contains the essential informationregarding each feature. Format:

            >>> {'numeric': {'feature name' :
                            {
                                'index' : <index of feature column>,
                                'min' : <min value>,
                                'max' : <max value>,
                                'mean': <mean value>,
                                'std': <standard deviation>,
                                'median': <median value>,
                                'q1': <first quartile of the distribution>,
                                'q3': <third quartile of the distribution,
                            },
                        ...,
                        ...,
                        },
            'categorical: {'feature name':
                                {
                                    'index' : <index of feature column>,
                                    'distinct_values' : <distinct categorical values>,
                                    'value_counts' : {'distinct value' : <elements count>,
                                                    ... }
                                }
                            },
                            ...
                            ...
                            ...     
            }
    """
    def __init__(self,data: DataFrame, class_name:str = None):
        super().__init__()
        self.class_name = class_name
        self.df = data

        #target columns forced to be the last column of the dataset
        if class_name is not None:
            self.df = self.df[[x for x in self.df.columns if x != class_name] + [class_name]]
        
        self.descriptor = {'numeric':{}, 'categorical':{}}

        #creation of a default version of descriptor
        self.descriptor = self.update_descriptor(self.df)
        print(self.descriptor)
        
    def update_descriptor(self, df: DataFrame):
        """
        it creates the dataset descriptor dictionary
        """
        descriptor = {'numeric':{}, 'categorical':{}}
        for feature in df.columns:
            index = df.columns.get_loc(feature)
            if feature in df.select_dtypes(include=np.number).columns.tolist():
                #numerical
                desc = {'index': index,
                        'min' : df[feature].min(),
                        'max' : df[feature].max(),
                        'mean':df[feature].mean(),
                        'std':df[feature].std(),
                        'median':df[feature].median(),
                        'q1':df[feature].quantile(0.25),
                        'q3':df[feature].quantile(0.75),
                        }
                descriptor['numeric'][feature] = desc
            else:
                #categorical feature
                desc = {'index': index,
                        'distinct_values' : list(df[feature].unique()),
                        'count' : {x : len(df[df[feature] == x]) for x in list(df[feature].unique())}}
                descriptor['categorical'][feature] = desc

        descriptor = self.set_target_label(descriptor)
        return descriptor


        
    
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
        logger.info('{0} file imported'.format(filename))
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

