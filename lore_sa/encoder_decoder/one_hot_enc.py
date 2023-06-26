from .enc_dec import EncDec
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from lore_sa.dataset.dataset import Dataset

__all__ = ["EncDec", "OneHotEnc"]


class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    It relies on OneHotEncoder class from sklearn
    """

    def __init__(self):
        super().__init__()
        self.encoder = OneHotEncoder()
        self.type='one-hot'

    def encode(self, dataset: Dataset, features_to_encode: list):
        """
        It applies the encoder to the input features. It also modifies the input dataset object, adding the
        encoded version of the dataset

        :param[Dataset] dataset: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        
        #tranforming everything to numpy, in order to not get everything messed by pandas indexing
        encoded = self.encoder.fit_transform(dataset.df[features_to_encode].to_numpy()).toarray()
        original = dataset.df.drop(columns=features_to_encode)
        self.original_features_skipped = original.columns
        original = original.to_numpy()

        dataset.dataset_encoded = pd.DataFrame(encoded.astype(int), columns = self.encoder.get_feature_names_out(dataset.df[features_to_encode].columns))
        # buffering original features
        self.original_data = dataset.df[features_to_encode]
   
        self.original_features_encoded = features_to_encode
        self.encoded_features = dataset.dataset_encoded.columns

        # substituing old features with encoded
        dataset.df = pd.concat([pd.DataFrame(original, columns = dataset.df.drop(columns=features_to_encode).columns), dataset.dataset_encoded], axis=1)

        return dataset.df.to_numpy()

    def __str__(self):
        if self.encoded_features is not None:
            return "OneHotEncoder - features encoded: %s" % (",".join(self.original_features_encoded))
        else:
            return "OneHotEncoder - no features encoded"

    def decode(self, dataset: Dataset):
        """
        It decodes the input dataset using onehotencoder inverse_transform. The input dataset must be encoded, with the encoded
        part contained into the property dataset.dataset_encoded
        

        :param dataset: the Dataset containing the features to be encoded
        """

        if dataset.dataset_encoded is None:
            raise Exception("No encoded dataset found")
        
        else:
            #tranforming everything to numpy, in order to not get everything messed by pandas indexing

            decoded = self.encoder.inverse_transform(dataset.dataset_encoded.to_numpy())
            original = dataset.df.drop(columns=self.encoded_features).to_numpy()
            
                
            dataset.df = pd.concat([pd.DataFrame(original, columns = self.original_features_skipped),
                                     pd.DataFrame(decoded, columns = self.original_features_encoded)], axis=1)

            
            return dataset.df.to_numpy()
