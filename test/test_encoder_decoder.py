import unittest
from lore_sa.encoder_decoder import OneHotEnc, LabelEnc
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder.my_target_enc import TargetEnc


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
        dataset = TabularDataset.from_csv("resources/adult.csv")

        one_hot_enc = OneHotEnc()
        dataset_encoded = one_hot_enc.encode(dataset,['race','sex'])
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - features encoded: race,sex")

    def test_target_encoder_init_with_features_encoder(self):
        dataset = TabularDataset.from_csv("resources/adult.csv",class_name = "capital-gain")
        target_enc = TargetEnc()
        dataset_encoded = target_enc.encode(dataset, ['workclass','education','occupation','race','sex'], target = "capital-gain")
        self.assertEqual(target_enc.__str__(),"TargetEncoder - features encoded: workclass,education,occupation,race,sex - target feature: capital-gain")

    def test_decode_error(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']}, class_name='col3')
        one_hot_enc = OneHotEnc()
        with  self.assertRaises(Exception) as context:
            one_hot_enc.decode(dataset)
            self.assertEqual("ERROR! To decode a dataset it must be firstly encoded by the same encoder object.", str(context.exception))


    def test_label_encoder_init_with_features_encoder(self):
        dataset = TabularDataset.from_csv("resources/adult.csv")

        features_encoding = {'race': {0: 'Amer-Indian-Eskimo', 1: 'Asian-Pac-Islander', 2: 'Black', 3: 'Other', 4: 'White'}, 'sex': {0: 'Female', 1: 'Male'}}

        label_enc = LabelEnc()
        dataset_encoded = label_enc.encode(dataset,['race','sex'])
        self.assertEqual(label_enc.__str__(),"LabelEncoder - features encoded: race,sex")
        self.assertTrue('race' in dataset_encoded.columns )
        self.assertTrue(dataset_encoded['race'].all() in [0,6])
        self.assertTrue('sex' in dataset_encoded.columns )
        self.assertTrue(dataset_encoded['sex'].all() in [0,6])

    def test_label_decode(self):
        dataset = TabularDataset.from_csv("resources/adult.csv")
        features_encoding = {'race': {0: 'Amer-Indian-Eskimo', 1: 'Asian-Pac-Islander', 2: 'Black', 3: 'Other', 4: 'White'}, 'sex': {0: 'Female', 1: 'Male'}}
        label_enc = LabelEnc()
        dataset_encoded = TabularDataset(label_enc.encode(dataset,['race','sex']))
        dataset_decoded = label_enc.decode(dataset_encoded,features_encoding)

        self.assertTrue('sex' in dataset_decoded.columns)
        self.assertEqual(dataset_decoded['sex'].unique().tolist(),['Male', 'Female'])
        self.assertTrue('race' in dataset_decoded.columns)
        self.assertEqual(dataset_decoded['race'].unique().tolist(),['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

    def test_label_decode_without_feature_encoding(self):
        dataset = TabularDataset.from_csv("resources/adult.csv")
        label_enc = LabelEnc()
        dataset_encoded = TabularDataset(label_enc.encode(dataset,['race','sex']))
        dataset_decoded = label_enc.decode(dataset_encoded)

        self.assertTrue('sex' in dataset_decoded.columns)
        self.assertEqual(dataset_decoded['sex'].unique().tolist(),['Male', 'Female'])
        self.assertTrue('race' in dataset_decoded.columns)
        self.assertEqual(dataset_decoded['race'].unique().tolist(),['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])


if __name__ == '__main__':
    unittest.main()
