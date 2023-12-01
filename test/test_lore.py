import os
import pickle
import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import TabularEnc
from lore_sa.encoder_decoder.enc_dec import IdentityEncoder
from lore_sa.lore.lore import Lore
from lore_sa.neighgen import TabularRandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate


class LoremTest(unittest.TestCase):

    def setUp(self):
        dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        dataset.df.dropna(inplace=True)
        dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        dataset.update_descriptor()
        encoder = TabularEnc(dataset.descriptor)

        model_pkl_file = "resources/adult_random_forest.pkl"
        if os.path.exists(model_pkl_file):
            with open(model_pkl_file, 'rb') as f:
                bb = pickle.load(f)
        else:
            encoded = []
            for x in dataset.df.iloc:
                encoded.append(encoder.encode(x.values))

            ext_dataset = pd.DataFrame(encoded, columns=[encoder.encoded_features[i] for i in range(len(encoded[0]))])
            feature_names = [c for c in ext_dataset.columns if c != dataset.class_name]
            class_name = dataset.class_name
            test_size = 0.3
            random_state = 42

            X_train, X_test, y_train, y_test = train_test_split(ext_dataset[feature_names].values,
                                                                ext_dataset[class_name].values, test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=ext_dataset[class_name].values)
            bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
            bb.fit(X_train, y_train)
            with open(model_pkl_file, 'wb') as f:
                pickle.dump(bb, f)



        self.bbox = sklearn_classifier_bbox.sklearnBBox(bb)
        self.gen = TabularRandomGenerator()
        self.dataset = dataset
        self.surrogate = DecisionTreeSurrogate()
        self.encoder = encoder


        # when

        # then
        pass

    def test_lorem_init(self):
        # given
        lore = Lore(self.bbox, self.dataset, IdentityEncoder(self.encoder.encoded_descriptor), neighgen=self.gen, surrogate=self.surrogate)
        self.assertIsInstance(lore, Lore)
        # when

        # then
        pass

    def test_explain_instance(self):
        # given
        lore = Lore(self.bbox, self.dataset, IdentityEncoder(self.encoder.encoded_descriptor), neighgen=self.gen, surrogate=self.surrogate)
        num_row = 10
        x = self.encoder.encode(self.dataset.df.iloc[num_row].values)
        # when
        exp = lore.explain(x)


        # then
        pass



if __name__ == '__main__':
    unittest.main()
