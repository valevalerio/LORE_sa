from lore_sa.bbox.bbox import AbstractBBox

__all__ = ["AbstractBBox", "keras_classifier_wrapper"]

class keras_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)