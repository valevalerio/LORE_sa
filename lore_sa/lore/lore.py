from lore_sa.bbox import AbstractBBox
import pandas as pd
import numpy as np

from lore_sa.dataset import Dataset
from lore_sa.encoder_decoder import EncDec
from lore_sa.neighgen.random import NeighborhoodGenerator
from lore_sa.surrogate import Surrogate

class Lore(object):

    def __init__(self, bbox: AbstractBBox,
                 dataset: Dataset,
                 encoder: EncDec,
                 neighgen: NeighborhoodGenerator,
                 surrogate: Surrogate):
        """
        LOREM Explainer, general setup for all the explainers
        Parameters
        ----------
        bbox [AbstractBBox]: the black box model to be explained
        dataset [Dataset]: the dataset with the descriptor of the content of the data
        encoder [EncDec]: the encoder to be used to encode/decode the data
        neighgen [NeighborhoodGenerator]: the neighborhood generator to be used to generate the neighborhood starting
        from a given instance
        """
        super().__init__()
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.neighgen = neighgen
        self.surrogate = surrogate

        self.features = [c for c in self.dataset.df.columns if c != self.dataset.class_name]
        self.class_name = self.dataset.class_name


    def explain(self, x: np.array, neighborhood_size: int = 1000):
        # generation of the neighborhood
        z = self.encoder.encode(x)
        neighbourhood = self.neighgen.generate(z, neighborhood_size, self.dataset.descriptor, onehotencoder=self.encoder)

        decoded_neighborhood = self.encoder.decode(neighbourhood.df[self.features].values)
        # supervising neighborhood instances with the black box
        neighbourhood_classes = self.bbox.predict(decoded_neighborhood)

        # learn a surrogate model from the neighborhood
        surrogate_model = self.surrogate.train(neighbourhood, neighbourhood_classes)

        rule = self.surrogate.get_rule(z, surrogate_model, neighbourhood, self.encoder)
        crules, self.deltas = surrogate_model.get_counterfactual_rules(x=z, class_name=self.class_name,
                                                                      feature_names=self.features,
                                                                      neighborhood_dataset=neighbourhood,
                                                                      encoder=self.encoder)

        return {
            'rule': rule,
            'counterfactual_rules': crules
        }

