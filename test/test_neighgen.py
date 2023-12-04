import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import TabularEnc
from lore_sa.logger import logger
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator


class NeighgenTest(unittest.TestCase):

    def setUp(self):
        self.dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        self.dataset.descriptor = self.dataset.update_descriptor(self.dataset.df)


    def test_random_generator(self):
        gen = RandomGenerator(bbox=None, dataset=self.dataset, encoder=None, ocr=0.1)
        self.assertIsInstance(gen, NeighborhoodGenerator)
        self.assertIsInstance(gen, RandomGenerator)

    def test_random_generator_generate_balanced(self):
        enc = TabularEnc(self.dataset.descriptor, self.dataset.class_name)
        encoded = []
        for v in self.dataset.df.iloc:
            encoded.append(enc.encode(v.values))
        encoded_dataset = pd.DataFrame(encoded, columns=[enc.encoded_features[i] for i in range(len(encoded[0]))])
        feature_names = [c for c in encoded_dataset.columns if c != self.dataset.class_name]
        class_name = self.dataset.class_name
        test_size = 0.3
        random_state = 42

        X_train, X_test, y_train, y_test = train_test_split(encoded_dataset[feature_names].values,
                                                            encoded_dataset[class_name].values, test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=encoded_dataset[class_name].values)

        bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
        bb.fit(X_train, y_train)
        Y_pred = bb.predict(X_test)

        logger.info('Accuracy: {}'.format(accuracy_score(y_test, Y_pred)))
        logger.info('F1: {}'.format(f1_score(y_test, Y_pred)))
        bbox = sklearn_classifier_bbox.sklearnBBox(bb)

        num_row = 10
        z = encoded_dataset.iloc[num_row][feature_names].values


        gen = RandomGenerator(bbox=bbox, dataset=self.dataset, encoder=enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, onehotencoder=enc)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.shape[0], 1000)

    def test_random_generator_generate_raw(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        enc = TabularEnc(self.dataset.descriptor)
        z = enc.encode(x.values)

        gen = RandomGenerator(bbox=None, dataset=self.dataset, encoder=enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, onehotencoder=enc)
        self.assertEqual(neighbour.shape[0], 1000)
        self.assertEqual(neighbour.shape[1], len(z))





if __name__ == '__main__':
    unittest.main()
