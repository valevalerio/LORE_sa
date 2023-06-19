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
    def generate(self, x, num_instances, features_domain = {}):
        """
        It generates similar instances 

        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param features_domain[dict]: dictionary in the format {"feature name": List | Range}, representing the domain range for each feature
        """
        return

    @abstractmethod
    def check_generated(self, filter_function = None, check_fuction = None):
        """
        It contains the logic to check the requirements for generated data
        """
        return
        
    
