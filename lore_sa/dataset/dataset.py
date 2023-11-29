__all__ = ["Dataset"]

from abc import abstractmethod


class Dataset():
    """
    Generic class to handle datasets
    """
    @abstractmethod
    def update_descriptor(self):
        """
        it updates the dataset descriptor dictionary depending on the content of the dataset
        """

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def get_feature_name(self, index):
        pass