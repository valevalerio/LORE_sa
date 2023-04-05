import unittest
from lore_sa.dataset import DataSet

class DatasetTest(unittest.TestCase):

    def test_dataset_class_int(self):
        # when
        dataset = DataSet('filename', ['class_one','class_two'])
        # then
        self.assertIs(dataset.encdec,None)



if __name__ == '__main__':
    unittest.main()
