from abc import abstractmethod
import numpy as np
import warnings

from lore_sa.encoder_decoder import MyTargetEnc, OneHotEnc

warnings.filterwarnings("ignore")

__all__ = ["NeighborhoodGenerator"]

class NeighborhoodGenerator(object):
    """This is a class generator
    """

    def __init__(self, bb_predict=None,  bb_predict_proba=None, feature_values=None, features_map=None, nbr_features=None, nbr_real_features=None,
                 numeric_columns_index=None, ocr=0.1, encdec=None, original_data=None):
        """

        :param bb_predict:
        :param bb_predict_proba:
        :param feature_values:
        :param features_map:
        :param nbr_features:
        :param nbr_real_features:
        :param numeric_columns_index:
        :param ocr:
        :param encdec:
        :param original_data:
        """
        self.bb_predict = bb_predict

        self.bb_predict_proba = bb_predict_proba
        self.feature_values = feature_values
        self.features_map = features_map
        self.nbr_features = nbr_features
        self.nbr_real_features = nbr_real_features
        self.numeric_columns_index = numeric_columns_index
        self.ocr = ocr  # other class ratio
        self.encdec = encdec
        self.original_data = original_data

    @abstractmethod
    def generate(self, x, num_samples=1000):
        """
         Generates `num_samples' synthetic records starting from the orginal value `x'

         x: Any, record instance to use as seed
         num_sample: int, the size of the neighborhood to generate
        """
        return

    def multi_generate(self, x, samples=1000, runs=1):
        Z_list = list()
        for i in range(runs):
            # if self.verbose:
            #     print('generating neighborhood [%s/%s] - %s' % (i, runs, self.neigh_gen.__class__))
                #print(samples, x)
            Z = self.generate(x, samples)
            Z_list.append(Z)
        return Z_list

    # qui dobbiamo prima decodificare
    def apply_bb_predict(self, X, encoded = None):
        if self.encdec is not None and encoded == None:
            X = self.encdec.dec(X)
        return self.bb_predict(X)

    # qui dobbiamo prima decodificare
    def apply_bb_predict_proba(self, X, encoded = None):
        if self.encdec is not None and encoded == None:
            X = self.encdec.dec(X)
        return self.bb_predict_proba(X)

    #TODO: sistemare per target encoding
    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):
        z = np.zeros(self.nbr_features) if from_z is None else from_z
        for i in range(self.nbr_real_features):
            if np.random.random() <= mutpb:
                real_feature_value = np.random.choice(self.feature_values[i], size=1, replace=True)
                if i in self.numeric_columns_index:
                    z[i] = real_feature_value
                elif type(self.encdec) is OneHotEnc:
                    idx = self.features_map[i][real_feature_value[0]]
                    z[idx] = 1.0
                elif type(self.encdec) is MyTargetEnc:
                    encs = self.encdec.get_cate_map(i, real_feature_value[0])
                    z[i] = encs
        return z

    def balance_neigh(self, x, Z, num_samples):
        Yb = self.apply_bb_predict(Z)
        #Yb = self.bb_predict(Z)
        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            #Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x)[0])
            Z1 = self.__rndgen_not_class(ocs, self.apply_bb_predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(ocs, self.apply_bb_predict(x.reshape(1, -1))[0])
                #Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x)[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z = self.generate_synthetic_instance()
            y = self.apply_bb_predict(z.reshape(1, -1))[0]
            #y = self.bb_predict(z)[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        Z = np.nan_to_num(Z)
        return Z
#qui arriva il dato gia passato nel decoder

