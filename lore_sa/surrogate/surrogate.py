from abc import ABC, abstractmethod

__all__ = ["Surrogate"]

class Surrogate(ABC):
    """
    Generic surrogate class
    """
    def __init__(self, kind = None, preprocessing =None):
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

    @abstractmethod
    def train(self, Z, Yb, weights):
        pass

