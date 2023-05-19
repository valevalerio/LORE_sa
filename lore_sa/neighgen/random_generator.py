from lore_sa.neighgen.neighgen import NeighborhoodGenerator
import numpy as np

__all__ = ["NeighborhoodGenerator","RandomGenerator"]

class RandomGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, encdec=None):
        super(RandomGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                              nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                              numeric_columns_index= numeric_columns_index, ocr=ocr, encdec=encdec)

    def generate(self, x, num_samples=1000):
        Z = np.zeros((num_samples, self.nbr_features))
        for j in range(num_samples):
            Z[j] = self.generate_synthetic_instance()

        Z = super(RandomGenerator, self).balance_neigh(x, Z, num_samples)
        Z = np.nan_to_num(Z)
        Z[0] = x.copy()
        return Z