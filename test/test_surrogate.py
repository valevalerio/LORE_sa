import os
import unittest

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate

class SurrogateTest(unittest.TestCase):

    def setUp(self):
        self.dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        self.dataset.update_descriptor()
        # print('descriptor', self.dataset.descriptor)
        self.enc = ColumnTransformerEnc(self.dataset.descriptor)

        model_pkl_file = "resources/adult_random_forest.pkl"
        if os.path.exists(model_pkl_file):
            bb = joblib.load(model_pkl_file)
        else:
            # print("Training the model", self.dataset.df.values)
            encoded = self.enc.encode(self.dataset.df.values)
            # print('original', self.dataset.df.iloc[0].values)
            # print('encoded', encoded[0])
            # for x in self.dataset.df.iloc:
            #     encoded.append(self.enc.encode(x.values))

            ext_dataset = pd.DataFrame(encoded, columns=[self.enc.encoded_features[i] for i in range(len(encoded[0]))])
            # print(self.dataset.df.head())
            # print(ext_dataset.head())
            self.feature_names = [c for c in ext_dataset.columns if c != self.dataset.class_name]
            class_name = self.dataset.class_name
            test_size = 0.3
            random_state = 42

            X_train, X_test, y_train, y_test = train_test_split(ext_dataset[self.feature_names].values,
                                                                ext_dataset[class_name].values, test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=ext_dataset[class_name].values)

            bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
            bb.fit(X_train, [int(v) for v in y_train]) # it seems that RF needs this explicit cast. I do not know why!
            joblib.dump(bb, model_pkl_file)

        self.bbox = sklearn_classifier_bbox.sklearnBBox(bb)

    def test_init(self):
        iris = load_iris()
        dt = DecisionTreeSurrogate().train(iris.data, iris.target)
        self.assertIsInstance(dt, DecisionTreeClassifier)


    def test_fit_from_random_neighborhood(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        z = self.enc.encode([x.values])[0][:-1] # remove the class feature from the input instance

        gen = RandomGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)

        # split neighbor in features and class using train_test_split
        neighb_train_X = neighbour[:, :-1]
        neighb_train_y = neighbour[:, -1].astype(int) # cast to int because the classifier needs it. Why?

        dt = DecisionTreeSurrogate()
        dt.train(neighb_train_X, neighb_train_y)
        # print(export_text(dt.dt, feature_names=list(self.enc.encoded_features.values())[:-1]))
        rule = dt.get_rule(z, self.enc)
        print('rule', rule)
        crules, deltas = dt.get_counterfactual_rules(z, neighb_train_X, neighb_train_y, self.enc)
        print('\n crules')
        for c in crules:
            print(c)
        for d in deltas:
            print(d)

if __name__ == '__main__':
    unittest.main()
