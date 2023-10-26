import numpy as np

from lore_sa.bbox.bbox import AbstractBBox
from keras.models import model_from_json

__all__ = ["AbstractBBox", "keras_classifier_wrapper"]

class keras_classifier_wrapper(AbstractBBox):
    def __init__(self, model):
        super().__init__()
        self.bbox = model


    @classmethod
    def model_from_json(cls, path, model_name):
        model_filename = f"{path}{model_name}.json"
        weights_filename = f"{path}{model_name}_weights.hdf5"
        bb = model_from_json(open(model_filename, 'r').read())
        bb.load_weights(weights_filename)

        bb_wrapper = cls(bb)
        return bb_wrapper


    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = X.astype('float32') / 255.
        Y = self.bbox.predict(X)
        return Y