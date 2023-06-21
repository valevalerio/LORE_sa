import unittest
import pandas
import numpy as np

from lore_sa.dataset.dataset import Dataset

class DatasetTest(unittest.TestCase):


    def test_dataset_class_init_from_csv(self):
        # when
        dataset = Dataset.from_csv("resources/adult.csv")
        # then
        self.assertIs(dataset.filename,"resources/adult.csv")
        self.assertIsInstance(dataset.df, pandas.DataFrame)

    def test_dataset_class_init_from_dict(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4]})

        self.assertIsInstance(dataset.df, pandas.DataFrame)

    def test_dataset_set_class_name(self):
        name = "workclass"
        dataset = Dataset.from_csv("resources/adult.csv")
        dataset.set_class_name(name)
        self.assertEqual(dataset.class_name,name)

    def test_dataset_set_class_name_error_list(self):
        name = ["workclass","race"]
        dataset = Dataset.from_csv("resources/adult.csv")
        dataset.set_class_name(name)
        self.assertEqual(dataset.class_name,name)

    def test_dataset_set_class_name_initialization(self):
        name = "workclass"
        dataset = Dataset.from_csv("resources/adult.csv", class_name = name)
        self.assertEqual(dataset.class_name, name)

    def test_dataset_get_original_dataset(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4]})
        dataset.df.drop(columns=["col1"],inplace=True)

        self.assertEqual(len(dataset.df.columns), 1)
        self.assertEqual(len(dataset.get_original_dataset()),2)

    def test_get_numeric_columns(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America','Europe']})
        self.assertEqual(dataset.get_numeric_columns(),['col1','col2'])

    def test_get_class_value_as_exptected(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']},class_name='col3')
        self.assertEqual(dataset.get_class_values().tolist(),['America', 'Europe'])

    def test_get_class_value_raise_error(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']})
        with  self.assertRaises(Exception) as context:
            dataset.get_class_values()
            self.assertEqual("ERR: class_name is None. Set class_name with set_class_name('<column name>')", str(context.exception))

    def test_create_feature_map(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']})
        self.assertEqual(len(dataset.get_numeric_columns()),2)
        self.assertEqual(dataset.get_numeric_columns(),['col1','col2'])
        self.assertEqual(dataset.create_feature_map(),dict(numeric_columns=['col1','col2'], categorical_columns=['col3']))


if __name__ == '__main__':
    unittest.main()
