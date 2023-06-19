from lore_sa.neighgen.neighgen import NeighborhoodGenerator
import numpy as np

__all__ = ["NeighborhoodGenerator","RandomGenerator"]

class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """
    def __init__(self):
        self.generated_data = None


    def generate(self,x, num_instances, features_domain = {}):
        """
        random generation of new instances. The starting instance x is only used to detect the value type of each feature, in order
        to generate new values only for numeric features.
        
        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param features_domain[dict]: dictionary in the format {"feature name": List | Range}, representing the domain range for each feature. 
        The list (or range) associated to each key is used to randomly choice an element within the list. 
        """

        
        generated_list = []
        for n in num_instances:
            instance = {}
            for feature in x:
                feature_value = x[feature]
                if feature in features_domain: 
                    #feature_value is a list or a range
                    instance[feature] = np.random.choice(list(feature_value))
                elif isinstance(feature_value, float) or isinstance(feature_value, int):
                    
                    #no information on the dataset, generation of random instances only for the numeric features of the input real instance

                    cast = type(feature_value)
                    instance[feature] = cast(np.random.randn())
                else:
                    #feature is a string/category, it is impossibile to generate random values
                    instance[feature] = feature_value
            generated_list.append(instance)

        self.generated_data = generated_list

        return generated_list
                        
    
                    