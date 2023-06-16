import unittest
from lore_sa.encoder_decoder import OneHotEnc, TargetEnc
from lore_sa.dataset import Dataset


class EncDecTest(unittest.TestCase):

    def test_one_hot_encoder_init(self):
        one_hot_enc = OneHotEnc()
        self.assertEqual(one_hot_enc.type,'one-hot')
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - no features encoded")

    def test_target_encoder_init(self):
        target_enc = TargetEnc()
        self.assertEqual(target_enc.type,'target')
        self.assertEqual(target_enc.__str__(),"TargetEncoder - no features encoded")

    def test_one_hot_encoder_init_with_features_encoder(self):
        dataset = Dataset.from_csv("resources/adult.csv")
        one_hot_enc = OneHotEnc()
        dataset_encoded = one_hot_enc.encode(dataset,['race','sex'])
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - features encoded: race,sex")

    def test_target_encoder_init_with_features_encoder(self):
        dataset = Dataset.from_csv("resources/adult.csv",class_name = "capital-gain")
        target_enc = TargetEnc()
        dataset_encoded = target_enc.encode(dataset, ['workclass','education','occupation','race','sex'], target = "capital-gain")
        self.assertEqual(target_enc.__str__(),"TargetEncoder - features encoded: workclass,education,occupation,race,sex - target feature: capital-gain")

    def test_decode_error(self):
        dataset = Dataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']}, class_name='col3')
        one_hot_enc = OneHotEnc()
        with  self.assertRaises(Exception) as context:
            one_hot_enc.decode(dataset)
            self.assertEqual("ERROR! To decode a dataset it must be firstly encoded by the same encoder object.", str(context.exception))


if __name__ == '__main__':
    unittest.main()
