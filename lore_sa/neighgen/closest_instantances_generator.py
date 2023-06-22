from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.util import calculate_feature_values, neuclidean
from lore_sa.neighgen.random_generator import RandomGenerator

from scipy.spatial.distance import cdist
import numpy as np

__all__ = ["NeighborhoodGenerator","ClosestInstancesGenerator"]
class ClosestInstancesGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, K=None, rK=None, k=None, core_neigh_type='unified', alphaf=0.5,
                 alphal=0.5, metric_features=neuclidean, metric_labels='hamming', categorical_use_prob=True,
                 continuous_fun_estimation=False, size=1000, encdec=None, verbose=False):
        super(ClosestInstancesGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                        nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                        numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.K = K
        self.rK = rK
        self.k = k if k is not None else int(0.5 * np.sqrt(len(self.rK))) + 1
        # self.k = np.min([self.k, len(self.rK)])
        self.core_neigh_type = core_neigh_type
        self.alphaf = alphaf
        self.alphal = alphal
        self.metric_features = metric_features
        self.metric_labels = metric_labels
        self.categorical_use_prob = categorical_use_prob
        self.continuous_fun_estimation = continuous_fun_estimation
        self.size = size
        self.verbose = verbose

    def generate(self, x, num_samples=1000):

        K = np.concatenate((x.reshape(1, -1), self.K), axis=0)
        #Yb = self.bb_predict(K)
        Yb = self.apply_bb_predict(K)
        if self.core_neigh_type == 'mixed':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:max(int(self.k * self.alphaf), 2)] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]

            ldist = cdist(Yb, Yb[0].reshape(1, -1), metric=self.metric_labels).ravel()
            rk_idxs = np.where(np.argsort(ldist)[:max(int(self.k * self.alphal), 2)] < len(self.rK))[0]
            Zl = self.rK[rk_idxs]
            rZ = np.concatenate((Zf, Zl), axis=0)
        elif self.core_neigh_type == 'unified':
            def metric_unified(x, y):
                n = K.shape[1]
                m = Yb.shape[1]
                distf = cdist(x[:n].reshape(1, -1), y[:n].reshape(1, -1), metric=self.metric_features).ravel()
                distl = cdist(x[n:].reshape(1, -1), y[n:].reshape(1, -1), metric=self.metric_labels).ravel()
                return n / (n + m) * distf + m / (n + m) * distl
            U = np.concatenate((K, Yb), axis=1)
            Un = (U - np.min(U)) / (np.max(U) - np.min(U))
            udist = cdist(Un, Un[0].reshape(1, -1), metric=metric_unified).ravel()
            rk_idxs = np.where(np.argsort(udist)[:self.k] < len(self.rK))[0]
            rZ = self.rK[rk_idxs]
        else:  # self.core_neigh_type == 'simple':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:self.k] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]
            rZ = Zf

        if self.verbose:
            print('calculating feature values')

        feature_values = calculate_feature_values(rZ, self.numeric_columns_index,
                                                  categorical_use_prob=self.categorical_use_prob,
                                                  continuous_fun_estimation=self.continuous_fun_estimation,
                                                  size=self.size)
        rndgen = RandomGenerator(self.bb_predict, feature_values, self.features_map, self.nbr_features,
                                 self.nbr_real_features, self.numeric_columns_index, self.ocr)
        Z = rndgen.generate(x, num_samples)
        Z = np.nan_to_num(Z)
        return Z