import unittest

import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lore_sa.bbox import AbstractBBox, sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore


class LoremTest(unittest.TestCase):

    def setUp(self):
        # We load a toy dataset to be used in the tests
        data = sklearn.datasets.load_wine(as_frame=True)
        X = data['data']
        y = data['target']
        df = data['frame']
        df['target'] = df['target'].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        model.set_params(classifier__n_estimators=3).fit(X_train, y_train)
        print('model.score', model.score(X_test, y_test))
        # print confusion matrix of model
        print(confusion_matrix(y_test, model.predict(X_test)))

        bbox = sklearn_classifier_bbox.sklearnBBox(model)


        self.dataset = TabularDataset(data=data['frame'], class_name='target')
        print('dataset', self.dataset.descriptor)

        self.tabularLore = TabularRandomGeneratorLore(bbox, self.dataset)

    def test_lorem_init(self):
        # given

        # when

        # then
        pass

    def test_explain_instance(self):
        # given
        num_row = 10
        x = self.dataset.df.iloc[num_row]
        # when
        explanation = self.tabularLore.explain(x)
        # then
        print(explanation)
        pass



if __name__ == '__main__':
    unittest.main()
