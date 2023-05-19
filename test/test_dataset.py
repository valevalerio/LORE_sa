import unittest

import pandas

from lore_sa.dataset import DataSet

class DatasetTest(unittest.TestCase):


    def test_dataset_class_init_from_csv(self):
        # when
        dataset = DataSet.from_csv("resources/adult.csv")
        # then
        self.assertIs(dataset.filename,"resources/adult.csv")
        self.assertIsInstance(dataset.df, pandas.DataFrame)

    def test_dataset_class_init_from_dict(self):
        dataset = DataSet.from_dict({'col1': [1, 2], 'col2': [3, 4]})

        self.assertIsInstance(dataset.df, pandas.DataFrame)

if __name__ == '__main__':
    unittest.main()
