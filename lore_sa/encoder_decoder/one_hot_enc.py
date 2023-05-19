from .enc_dec import EncDec
import numpy as np

from sklearn.preprocessing import OneHotEncoder

__all__ = ["EncDec","OneHotEnc"]

extend: OneHotEncoder
class OneHotEnc(EncDec):
    def __init__(self, dataset, class_name):
        super(OneHotEnc, self).__init__(dataset, class_name)
        self.dataset_enc = None
        self.type="onehot"


    def enc(self, x, y, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.cate_features_idx]
            x_cat = x_cat.reshape(1, -1)
            x = x.reshape(1,-1)
        else:
            x_cat = x[:, self.cate_features_idx]

        x_cat_enc = self.encdec.transform(x_cat).toarray()
        n_feat_tot = self.dataset_enc.shape[1] + len(self.cont_features_idx)
        x_res = np.zeros((x.shape[0], n_feat_tot))
        for p in range(x_res.shape[0]):
            for i in range(0, len(self.onehot_feature_idx)):
                x_res[p][self.onehot_feature_idx[i]] = x_cat_enc[p][i]
            for j in range(0, len(self.new_cont_idx)):
                x_res[p][self.new_cont_idx[j]] = x[p][self.cont_features_idx[j]]
        return x_res


    def dec(self, x, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.onehot_feature_idx]
            x = x.reshape(1, -1)
            x_cat = x_cat.reshape(1, -1)
        else:
            x_cat = x[:, self.onehot_feature_idx]

        X_new = self.encdec.dec(x_cat)
        x_res = np.empty((x.shape[0], len(self.features)), dtype=object)
        for p in range(x.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = X_new[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = x[p][j]
        #print(x_res.shape)
        return x_res

    def enc_fit_transform(self, kwargs=None):
        """
        select the categorical variable
        apply onehot encoding on them
        self.encdec is the encoder already fitted
        self.dataset_enc is the dataset encoded

        :param kwargs:
        :return:
        """
        self.features = [c for c in self.dataset.columns if c not in [self.class_name]]
        self.cont_features_names = list(self.dataset[self.features]._get_numeric_data().columns)
        self.cate_features_names = [c for c in self.dataset.columns if
                                    c not in self.cont_features_names and c != self.class_name]
        self.cate_features_idx = [self.features.index(f) for f in self.cate_features_names]
        self.cont_features_idx = [self.features.index(f) for f in self.cont_features_names]
        print('cate features idx ', self.cate_features_idx)
        dataset_values = self.dataset[self.features].values
        self.encdec = OneHotEncoder(handle_unknown='ignore')
        self.dataset_enc = self.encdec.fit_transform(dataset_values[:, self.cate_features_idx]).toarray()

        self.onehot_feature_idx = list()
        self.new_cont_idx = list()
        for f in self.cate_features_idx:
            uniques = len(np.unique(dataset_values[:, f]))
            for u in range(0,uniques):
                self.onehot_feature_idx.append(f+u)
        npiu = i = j = 0
        while j < len(self.cont_features_idx):
            if self.cont_features_idx[j] < self.cate_features_idx[i]:
                self.new_cont_idx.append(self.cont_features_idx[j] + npiu - 1)
            elif self.cont_features_idx[j] > self.cate_features_idx[i]:
                npiu += len(np.unique(dataset_values[:, self.cate_features_idx[i]]))
                self.new_cont_idx.append(self.cont_features_idx[j] + npiu - 1)
                i += 1
            j += 1
        n_feat_tot = self.dataset_enc.shape[1] + len(self.cont_features_idx)
        self.dataset_enc_complete = np.zeros((self.dataset_enc.shape[0], n_feat_tot))
        for p in range(self.dataset_enc.shape[0]):
            for i in range(0, len(self.onehot_feature_idx)):
                self.dataset_enc_complete[p][self.onehot_feature_idx[i]] = self.dataset_enc[p][i]
            for j in range(0, len(self.new_cont_idx)):
                self.dataset_enc_complete[p][self.new_cont_idx[j]] = dataset_values[p][self.cont_features_idx[j]]
        #print(len(self.onehot_feature_idx))
        #print(len(self.cate_features_idx))
        return self.dataset_enc_complete
