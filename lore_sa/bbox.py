from abc import ABC, abstractmethod

class AbstractBBox(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, sample_matrix: list):
        pass

    @abstractmethod
    def predict_proba(self, sample_matrix: list):
        pass