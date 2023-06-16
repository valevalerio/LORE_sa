import unittest
from lore_sa.encoder_decoder import OneHotEnc
from lore_sa.dataset import Dataset


class EncDecTest(unittest.TestCase):

    def test_one_hot_encoder_init(self):
        one_hot_enc = OneHotEnc()
        self.assertEqual(one_hot_enc.type,'one-hot')
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - no features encoded")

    def test_one_hot_encoder_init_with_features_encoder(self):
        dataset = Dataset.from_csv("resources/adult.csv")
        one_hot_enc = OneHotEnc()
        dataset_encoded = one_hot_enc.encode(dataset,['race','sex'])
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - features encoded: race,sex")


if __name__ == '__main__':
    unittest.main()
