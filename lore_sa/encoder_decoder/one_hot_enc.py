from collections import defaultdict

from .enc_dec import EncDec
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

__all__ = ["EncDec","OneHotEnc"]

extend: OneHotEncoder
class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    """
    def __init__(self, dataset, class_name):
        super(OneHotEnc, self).__init__(dataset, class_name)
        self.dataset_enc = None
        self.type= "onehot"


    def enc(self, x, kwargs=None):
        """
        :param [numpyArray] x: feature values to be encoded. It could be either a vector or a matrix. Each column represents the values 
        of a feature 
        """
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

    def prepare_dataset(self, df, class_name:str, numeric_columns: list, rdf: pd.DataFrame):
        # faccio il one hot encoding, tramite pd.get_dummies, di tutte le variabili diverse da class_name
        df_dummy_not_class_name = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')

        # faccio un encoding dei valori che assume la variabile target class_name
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        df_dummy_class_name = df[class_name].map(class_name_map)

        # unisco il tutto
        df_dummy = pd.concat([df_dummy_not_class_name, df_dummy_class_name], axis=1)
        df_dummy = df_dummy.reindex(df_dummy_not_class_name.index)

        feature_names = list(df_dummy_not_class_name.columns)
        class_values = sorted(class_name_map)

        real_feature_names = self.__get_real_feature_names(rdf, numeric_columns, class_name)
        features_map = self.__get_features_map(feature_names, real_feature_names)

        rdf = rdf[real_feature_names + [class_name]]

        return df_dummy, feature_names, features_map, numeric_columns, class_values, rdf, real_feature_names

    def __get_real_feature_names(self,rdf, numeric_columns: list, class_name: str):
        if isinstance(class_name, list):
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
        else:
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
        return real_feature_names

    def __get_features_map(slef,feature_names, real_feature_names):
        features_map = defaultdict(dict)
        i = 0
        j = 0

        while i < len(feature_names) and j < len(real_feature_names):
            if feature_names[i] == real_feature_names[j]:
                #print('in if ', feature_names[i], real_feature_names[j])
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
                j += 1

            elif feature_names[i].startswith(real_feature_names[j]):
                #print('in elif ', feature_names[i], real_feature_names[j])
                features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
                i += 1
            else:
                j += 1
        return features_map