from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator

from lore_sa.dataset.dataset import TabularDataset
import pandas as pd
import numpy as np

__all__ = ["NeighborhoodGenerator","RandomGenerator"]

class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """
    def __init__(self):
        self.generated_data = None


    def generate(self,x, num_instances, descriptor = None):
        """
        random generation of new instances. The starting instance x is only used to detect the value type of each feature, in order
        to generate new values only for numeric features.
        
        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        The list (or range) associated to each key is used to randomly choice an element within the list. 

        :return [Dataset]: a dataset instance with the new data generated
        """

        generated_list = []
        for n in range(num_instances):
            instance = {}
            for feature in x:
                feature_value = x[feature]
                if type(feature_value) ==float :
                    
                    #no information on the dataset, generation of random instances only for the numeric features of the input real instance

                    instance[feature] = (np.random.uniform(descriptor['numeric'][feature]['min'], descriptor['numeric'][feature]['max']))

                elif type(feature_value) == int:
                    instance[feature] = np.random.random_integers(descriptor['numeric'][feature]['min'], descriptor['numeric'][feature]['max'])

                else:
                    #feature is a string/category, it is impossibile to generate random values
                    instance[feature] = np.random.choice(descriptor['categoric'][feature]['distinct_values'])
            generated_list.append(instance)

        self.generated_data = generated_list

        
        return TabularDataset(pd.DataFrame(generated_list))
                        
    
                    