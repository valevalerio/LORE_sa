from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator

from lore_sa.dataset.tabular_dataset import TabularDataset
import pandas as pd
import numpy as np

__all__ = ["NeighborhoodGenerator","RandomGenerator"]

class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """
    def __init__(self):
        self.generated_data = None

    def generate(self,x, num_instances, descriptor, onehotencoder = None):
        """
        random generation of new instances. The starting instance x is only used to detect the value type of each feature, in order
        to generate new values only for numeric features.
        
        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        :param onehotencoder[EncDec]: the onehotencoder eventually to encode the instance
        The list (or range) associated to each key is used to randomly choice an element within the list. 

        :return [TabularDataset]: a tabular dataset instance with the new data generated
        """

        generated_list, columns = [],[]

        columns = [None for e in range(len(x))]
        for n in range(num_instances):
            instance = self.generate_synthetic_instance(from_z=x)
            generated_list.append(instance)

        Z = self.balance_neigh(x, generated_list, num_instances)
        self.generated_data = Z

        return TabularDataset(pd.DataFrame(Z, columns = columns))
                        
    
                    