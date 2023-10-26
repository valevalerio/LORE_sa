import unittest

from lore_sa.bbox.keras_classifier_wrapper import keras_classifier_wrapper


class ImageDatasetTest(unittest.TestCase):

    def test_loading_model_from_json(self):
        bbox = keras_classifier_wrapper.model_from_json("resources/models", "cifar10_DNN")
        self.assertIsNotNone(bbox)



if __name__ == '__main__':
    unittest.main()