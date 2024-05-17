import os
import joblib
import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import TabularEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.genetic import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator


class NeighgenTest(unittest.TestCase):

    def setUp(self):
        self.dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        self.dataset.update_descriptor()
        self.enc = TabularEnc(self.dataset.descriptor)

        model_pkl_file = "resources/adult_random_forest.pkl"
        if os.path.exists(model_pkl_file):
            bb = joblib.load(model_pkl_file)
        else:
            encoded = []
            for x in self.dataset.df.iloc:
                encoded.append(self.enc.encode(x.values))

            ext_dataset = pd.DataFrame(encoded, columns=[self.enc.encoded_features[i] for i in range(len(encoded[0]))])
            self.feature_names = [c for c in ext_dataset.columns if c != self.dataset.class_name]
            class_name = self.dataset.class_name
            test_size = 0.3
            random_state = 42

            X_train, X_test, y_train, y_test = train_test_split(ext_dataset[self.feature_names].values,
                                                                ext_dataset[class_name].values, test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=ext_dataset[class_name].values)
            bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
            bb.fit(X_train, y_train)
            joblib.dump(bb, model_pkl_file)

        self.bbox = sklearn_classifier_bbox.sklearnBBox(bb)

    def test_random_generator(self):
        gen = RandomGenerator()
        self.assertIsInstance(gen, NeighborhoodGenerator)
        self.assertIsInstance(gen, RandomGenerator)

    def test_random_generator_generate_balanced(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        z = self.enc.encode(x.values)[:-1] # remove the class feature from the input instance

        gen = RandomGenerator()
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.df.shape[0], 1000)

    def test_random_generator_generate_raw(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        z = self.enc.encode(x.values)[:-1] # remove the class feature from the input instance

        gen = RandomGenerator()
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)
        self.assertEqual(neighbour.df.shape[0], 1000)
        self.assertEqual(neighbour.df.shape[1], len(z))


    def test_genetic_generator(self):
        gen = GeneticGenerator()
        self.assertIsInstance(gen, NeighborhoodGenerator)
        self.assertIsInstance(gen, GeneticGenerator)

    def test_genetic_generator_generate_balanced(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        z = self.enc.encode(x.values)[:-1] # remove the class feature from the input instance

        gen = GeneticGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1, ngen=20)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.shape[0], 1000)



if __name__ == '__main__':
    unittest.main()