from .enc_dec import EncDec
from category_encoders import TargetEncoder
from lore_sa.dataset.dataset import Dataset
import pandas as pd

__all__ = ["EncDec", "TargetEnc"]

extend: TargetEncoder
class TargetEnc(EncDec):
    """
    Target encoding for categorical features.
    Extend TargetEncoder from category_encoders.
    """
    def __init__(self):
        super().__init__()
        self.type = 'target'

    def encode(self, dataset: Dataset, features_to_encode: list, target):
        """
        It applies the encoder to the input categorical features

        :param[Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        :param[str] target: target name column
        """
        self.target = target

        y = dataset.df[target].values
        self.dataset_encoded = TargetEncoder(return_df=True, cols = features_to_encode).fit_transform(dataset.df,y)

        # buffering original features to get them eventually decoded
        self.original_data = dataset.df
        self.original_features = features_to_encode
        self.encoded_features = self.dataset_encoded.columns
        

        # substituing old features with encoded
        dataset.df = pd.concat([dataset.df.drop(columns=features_to_encode), self.dataset_encoded], axis=1)

        return dataset.df

    def __str__(self):
        if self.encoded_features is not None and self.target is not None:
            return "TargetEncoder - features encoded: {0} - target feature: {1}".format(",".join(self.original_features),self.target)
        else:
            return "TargetEncoder - no features encoded"

    def decode(self, dataset: Dataset, kwargs=None):
        if self.encoded_features is not None:
            if self.original_data is None:
                raise Exception("ERROR! To decode a dataset it must be firstly encoded by the same encoder object.")

            decoded_data = self.original_data
            self.dataset_encoded = None
            self.original_features = None
            self.original_data = None
            self.encoded_features = None        
            return decoded_data
        else:
            return dataset.df