import unittest

from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import OneHotEnc, LabelEnc
from lore_sa.encoder_decoder.my_target_enc import TargetEnc
import numpy as np

from lore_sa.encoder_decoder.tabular_encoder import TabularEnc

class EncDecTest(unittest.TestCase):

    descriptor_dummy = {}
    def setUp(self):
        self.descriptor_dummy = {'categoric': {'col3': {'count': {'America': 1, 'Europe': 1, 'Africa': 1},
                                                   'distinct_values': ['America', 'Europe', 'Africa'],
                                                   'index': 2},
                                          'colours': {
                                              'distinct_values': ['White', 'Black', 'Red', 'Blue', 'Green'],
                                              'index': 4
                                          }},
                            'numeric': {'col1': {'index': 0,'max': 2,'mean': 1.5,'median': 1.5,'min': 1,'q1': 1.25,
                                                 'q3': 1.75,'std': 0.7071067811865476},
                                        'col2': {'index': 1,'max': 4,'mean': 3.5,'median': 3.5,'min': 3,'q1': 3.25,
                                                 'q3': 3.75,'std': 0.7071067811865476}},
                            'ordinal': {'education': {'index': 3,
                                                      'distinct_values': ['Elementary', 'High School', 'College',
                                                                          'Graduate', 'Post-graduate']}
                                        }}

    def test_one_hot_encoder_init(self):
        one_hot_enc = OneHotEnc(self.descriptor_dummy)
        self.assertEqual(one_hot_enc.type,'one-hot')
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - no features encoded")

    def test_target_encoder_init(self):
        target_enc = TargetEnc(self.descriptor_dummy)
        self.assertEqual(target_enc.type,'target')
        self.assertEqual(target_enc.__str__(),"TargetEncoder - no features encoded")


    def test_one_hot_init_no_ordinal_field(self):
        wrong_descriptor = {}
        with self.assertRaises(Exception) as context:
            one_hot_enc = OneHotEnc(wrong_descriptor)
            self.assertEqual("Dataset descriptor is malformed for One-Hot Encoder: 'categoric' key is not present", str(context.exception))

    def test_one_hot_encoder_init_with_features_encoder(self):
        one_hot_enc = OneHotEnc(self.descriptor_dummy)
        encoded = one_hot_enc.encode(np.array([1, 2, "Europe", "Graduate", "Green"]))
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - features encoded: col3=America,col3=Europe,col3=Africa,colours=White,colours=Black,colours=Red,colours=Blue,colours=Green")
        self.assertEqual(encoded.tolist(),np.array([1, 2, 0, 1, 0, "Graduate", 0, 0, 0, 0, 1]).tolist())
        self.assertEqual(one_hot_enc.encoded_descriptor['numeric']['col1']['index'], 0)
        self.assertEqual(one_hot_enc.encoded_descriptor['numeric']['col2']['index'], 1)
        self.assertEqual(one_hot_enc.encoded_descriptor['ordinal']['education']['index'], 5)
        self.assertEqual(one_hot_enc.get_encoded_features(), {0:'col1',
                                                              1:'col2',
                                                              2:'col3=America',
                                                              3:'col3=Europe',
                                                              4:'col3=Africa',
                                                              5:'education',
                                                              6:'colours=White',
                                                              7:'colours=Black',
                                                              8:'colours=Red',
                                                              9:'colours=Blue',
                                                              10:'colours=Green'})

    def test_get_encoded_features_name_none(self):
        with self.assertRaises(Exception) as context:
            one_hot_enc = OneHotEnc(self.descriptor_dummy)
            one_hot_enc.get_encoded_features()
            self.assertEqual("You have not run the encoder yet", str(context.exception))

    def test_one_hot_decode_init_with_features_encoder(self):
        one_hot_enc = OneHotEnc(self.descriptor_dummy)
        decoded = one_hot_enc.decode(np.array([1, 2, 0, 0, 1, "Graduate", 0, 0, 0, 1, 0]))

        self.assertEqual(decoded.tolist(), np.array([1, 2, "Africa", "Graduate", "Blue"]).tolist())

    def test_label_init_no_ordinal_field(self):
        wrong_descriptor = {}
        with self.assertRaises(Exception) as context:
            label_enc = LabelEnc(wrong_descriptor)
            self.assertEqual("Dataset descriptor is malformed for Label Encoder: 'ordinal' key is not present", str(context.exception))


    def test_label_encoder_init_with_features_encoder(self):
        label_enc = LabelEnc(self.descriptor_dummy)
        encoded = label_enc.encode(np.array([1,2,"Europe","Graduate"]))
        self.assertEqual(label_enc.__str__(),"LabelEncoder - features encoded: education")
        self.assertEqual(encoded[3],'3')
        self.assertEqual(encoded.tolist(),np.array([1,2,"Europe",3]).tolist())
        self.assertEqual(label_enc.get_encoded_features(),{0: 'col1',
                                                           1: 'col2',
                                                           2: 'col3',
                                                           3: 'education',
                                                           4: 'colours'})


    def test_label_decode(self):
        label_enc = LabelEnc(self.descriptor_dummy)
        decoded = label_enc.decode(np.array([1,2,"Europe",0]))

        self.assertEqual(decoded[3],'Elementary')
        self.assertEqual(decoded.tolist(),np.array([1,2,"Europe",'Elementary']).tolist())


    def test_tabular_encode(self):
        tabular_enc = TabularEnc(self.descriptor_dummy)
        encoded = tabular_enc.encode(np.array([1, 2, "Europe", "Graduate", "Green"]))
        self.assertEqual(encoded.tolist(), np.array([1, 2, 0, 1, 0, 3, 0, 0, 0, 0, 1]).tolist())

    def test_tabular_encode_get_encoded_feature(self):
        tabular_enc = TabularEnc(self.descriptor_dummy)
        encoded = tabular_enc.encode(np.array([1, 2, "Europe", "Graduate", "Green"]))
        self.assertEqual(tabular_enc.get_encoded_features(),{0:'col1',
                                                             1:'col2',
                                                             2:'col3=America',
                                                             3:'col3=Europe',
                                                             4:'col3=Africa',
                                                             5:'education',
                                                             6:'colours=White',
                                                             7:'colours=Black',
                                                             8:'colours=Red',
                                                             9:'colours=Blue',
                                                             10:'colours=Green'})

    def test_tabular_encode_get_encoded_feature_with_target(self):
        tabular_enc = TabularEnc(self.descriptor_dummy, "education")
        encoded = tabular_enc.encode(np.array([1, 2, "Europe", "Graduate", "Green"]))
        self.assertEqual(tabular_enc.get_encoded_features(),{0:'col1',
                                                             1:'col2',
                                                             2:'col3=America',
                                                             3:'col3=Europe',
                                                             4:'col3=Africa',
                                                             5:'education',
                                                             6:'colours=White',
                                                             7:'colours=Black',
                                                             8:'colours=Red',
                                                             9:'colours=Blue',
                                                             10:'colours=Green'})

    def test_tabular_decode(self):
        tabular_enc = TabularEnc(self.descriptor_dummy)
        decoded = tabular_enc.decode(np.array([1, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0]))
        self.assertEqual(decoded.tolist(), np.array([1, 2, "Africa", "High School", 'White']).tolist())

    def test_tabular_decode_with_target(self):
        tabular_enc = TabularEnc(self.descriptor_dummy,"education")
        decoded = tabular_enc.decode(np.array([1, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0]))
        self.assertEqual(decoded.tolist(), np.array([1, 2, "Africa", "High School", 'White']).tolist())


if __name__ == '__main__':
    unittest.main()
