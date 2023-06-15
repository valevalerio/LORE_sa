from .enc_dec import EncDec
from category_encoders import TargetEncoder
from lore_sa.dataset.dataset import Dataset
import pandas as pd

__all__ = ["EncDec", "MyTargetEnc"]

extend: TargetEncoder
class MyTargetEnc(EncDec):
    """
    Target encoding for categorical features.
    Extend TargetEncoder from category_encoders.
    """
    def __init__(self):
        super().__init__()
        self.type = "target"


    def encode(self, dataset: Dataset, features_to_encode: list, target: str):
        """
        It applies the encoder to the input categorical features

        :param[Dataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        :param[str] target: target name column
        """
        y = dataset.df[target].values
        self.dataset_encoded = TargetEncoder(return_df=False).fit_transform(dataset.df,y)

        # buffering original features to get them eventually decoded
        self.original_data = dataset.df[features_to_encode]
        self.original_features = features_to_encode
        self.encoded_features = self.dataset_encoded.columns

        # substituing old features with encoded
        dataset.df = pd.concat([dataset.df.drop(columns=features_to_encode), self.dataset_encoded], axis=1)

        return dataset.df

    def __str__(self):
        if self.encoded_features is not None:
            return ("TargetEncoder - features encoded: %s"%(",".join(self.original_features)))
        else:
            return ("TargetEncoder - no features encoded")

    def decode(self, dataset: Dataset, kwargs=None):
        if self.encoded_features is not None:
            return pd.concat([dataset.df.drop(columns =self.encoded_features), self.original_data] , axis =1)
        else:
            return dataset.df