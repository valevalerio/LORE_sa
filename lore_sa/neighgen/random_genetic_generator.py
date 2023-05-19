from lore_sa.neighgen.genetic_generator import GeneticGenerator
from lore_sa.neighgen.random_generator import RandomGenerator
from lore_sa.util import neuclidean

import numpy as np

__all__ = ["GeneticGenerator","RandomGenerator","RandomGeneticGenerator"]

class RandomGeneticGenerator(GeneticGenerator, RandomGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, encdec=None, verbose=False):
        super(RandomGeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                     nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                     numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        super(RandomGeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                     nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                     numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec,
                                                     alpha1=alpha1, alpha2=alpha2, metric=metric, ngen=ngen, mutpb=mutpb,
                                                     cxpb=cxpb, tournsize=tournsize,
                                                     halloffame_ratio=halloffame_ratio, random_seed=random_seed, verbose=verbose)

    def generate(self, x, num_samples=1000):
        Zg = GeneticGenerator.generate(self, x, num_samples // 2)
        Zr = RandomGenerator.generate(self, x, num_samples // 2)
        Z = np.concatenate((Zg, Zr[1:]), axis=0)
        Z = np.nan_to_num(Z)
        return Z