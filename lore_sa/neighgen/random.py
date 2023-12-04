from abc import ABC

from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator

from lore_sa.dataset.tabular_dataset import TabularDataset
import pandas as pd
import numpy as np

__all__ = ["NeighborhoodGenerator", "RandomGenerator"]

class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """

    def __init__(self, bbox=None, dataset=None, encoder=None, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)
        self.generated_data = None

    def generate(self, x, num_instances, descriptor, onehotencoder=None):
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

        Z, columns = [], []


        for n in range(num_instances):
            instance = self.generate_synthetic_instance(from_z=x)
            Z.append(instance)

        if self.bbox is not None:
            Z = self.balance_neigh(x, Z, num_instances)
        self.generated_data = Z

        return TabularDataset(pd.DataFrame(Z, columns=self.columns))
