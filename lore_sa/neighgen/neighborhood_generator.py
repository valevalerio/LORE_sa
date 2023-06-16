from abc import abstractmethod
import numpy as np
import warnings


warnings.filterwarnings("ignore")

__all__ = ["NeighborhoodGenerator"]

class NeighborhoodGenerator(object):
    """
    Abstract class for Neighborhood generator. It defines the basic
    """

    @abstractmethod
    def __init__(self):
        self.generated_data = None 
        return
    
    @abstractmethod
    def generate(self):
        return

    @abstractmethod
    def check_generated(self, filter_function = None, check_fuction = None):
        """
        It contains the logic to check the requirements for generated data
        """

        
    
