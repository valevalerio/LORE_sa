import pandas as pd
import numpy as np

from .surrogate import DecisionTreeSurrogate, Surrogate
from .bbox import AbstractBBox
from .dataset import TabularDataset, Dataset
from .encoder_decoder import ColumnTransformerEnc, EncDec
from .neighgen.genetic import GeneticGenerator
from .neighgen.neighborhood_generator import NeighborhoodGenerator
from .neighgen.random import RandomGenerator
from .neighgen.genetic_proba_generator import GeneticProbaGenerator


class Lore(object):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec,
                 generator: NeighborhoodGenerator, surrogate: Surrogate):
        """
        Creates a new instance of the LORE method.


        :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
        :param dataset:
        :param encoder:
        :param generator:
        :param surrogate:
        """

        super().__init__()
        self.bbox = bbox
        self.descriptor = dataset.descriptor
        self.encoder = encoder
        self.generator = generator
        self.surrogate = surrogate
        self.class_name = dataset.class_name


    def explain(self, x: np.array, num_instances=1000):
        """
        Explains a single instance of the dataset.
        :param x: an array with the values of the instance to explain (the target class is not included)
        :return:
        """
        # map the single record in input to the encoded space
        [z] = self.encoder.encode([x])
        # generate a neighborhood of instances around the projected instance `z`
        neighbour = self.generator.generate(z.copy(), num_instances, self.descriptor, self.encoder)
        dec_neighbor = self.encoder.decode(neighbour)
        # split neighbor in features and class using train_test_split
        neighb_train_X = dec_neighbor[:, :]
        neighb_train_y = self.bbox.predict(neighb_train_X)
        neighb_train_yb = self.encoder.encode_target_class(neighb_train_y.reshape(-1, 1)).squeeze()

        # train the surrogate model on the neighborhood
        # this surrogate could be another model. I would love to try with apriori 
        # or the modified version of SAME (Single tree Approximation MEthod <3 )
        self.surrogate.train(neighbour, neighb_train_yb)

        # get the rule for the instance `z`, decode using the encoder class
        rule = self.surrogate.get_rule(z, self.encoder)
        # print('rule', rule)

        crules, deltas = self.surrogate.get_counterfactual_rules(z, neighbour, neighb_train_yb, self.encoder)

        return {
            # 'x': x.tolist(),
            'rule': rule.to_dict(),
            'counterfactuals': [c.to_dict() for c in crules],
            'fidelity': self.surrogate.fidelity
        }



class TabularRandomGeneratorLore(Lore):

    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
        Creates a new instance of the LORE method.
        :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
        :param dataset:
        :param encoder:
        :param generator:
        :param surrogate:
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = RandomGenerator(bbox, dataset, encoder, 0.1) # the last parameter is the ocr
        surrogate = DecisionTreeSurrogate()

        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array):
        return self.explain(x.values)

class TabularGeneticGeneratorLore(Lore):

    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
        Creates a new instance of the LORE method.


        :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
        :param dataset:
        :param encoder:
        :param generator:
        :param surrogate:
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticGenerator(bbox, dataset, encoder, 0.1)
        surrogate = DecisionTreeSurrogate()

        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array):
        return self.explain(x.values)
        
class TabularRandGenGeneratorLore(Lore):
     
    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
            Creates a new instance of the LORE method.
            :param bbox: The black box model to be explained wrapped in a ``AbstractBBox`` object.
            :param dataset:
            :param encoder:
            :param generator:
            :param surrogate:
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticProbaGenerator(bbox,
                                            dataset,
                                            encoder,
                                            0.1)
        surrogate = DecisionTreeSurrogate()
        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x:np.array):
        return self.explain(x.values)