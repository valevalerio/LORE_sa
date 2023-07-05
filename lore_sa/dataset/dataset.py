__all__ = ["Dataset"]

from abc import abstractmethod


class Dataset():
    """
    Generic class to handle datasets
    """
    @abstractmethod
    def update_descriptor(self):
        """
        it creates the dataset descriptor dictionary
        """

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor