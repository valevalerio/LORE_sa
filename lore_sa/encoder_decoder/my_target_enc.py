from .enc_dec import EncDec
from category_encoders import TargetEncoder
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["EncDec","MyTargetEnc"]

extend: TargetEncoder
class MyTargetEnc(EncDec):
    """
    Extend TargetEncoder from category_encoders

    """
    def __init__(self, dataset, class_name):
        super(MyTargetEnc, self).__init__(dataset, class_name)
        self.type = "target"
        self.dataset_enc = None
        self.cate_map = dict()
        self.inverse_cate_map = list()
        self.dataset_enc_complete = None


    def enc(self, x, y, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.cate_features_idx]
            x_cat = x_cat.reshape(1, -1)
            x = x.reshape(1,-1)
        else:
            x_cat = x[:, self.cate_features_idx]

        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        x_cat_enc = self.encdec.transform(x_cat, y)
        x_res = np.zeros((x.shape[0], x.shape[1]))
        for p in range(x.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = x_cat_enc[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = x[p][j]
        return x_res

    def dec(self, X, kwargs=None):
        if len(X.shape) == 1:
            X_cat = X[self.cate_features_idx]
            X = X.reshape(1, -1)
            X_cat = X_cat.reshape(1, -1)
        else:
            X_cat = X[:, self.cate_features_idx]

        X_cat = np.array(X_cat, dtype=float)
        X_new = list()
        for i in range(X_cat.shape[1]):
            values = np.array(list(self.inverse_cate_map[i].keys()))
            keys = np.array(list(self.inverse_cate_map[i].values()))
            closest_val = np.argmin(cdist(values.reshape(-1, 1), X_cat[:, i].reshape(-1, 1)), axis=0)
            X_new.append(np.array([keys[j] for j in closest_val]))
        X_new = np.array(X_new)
        # print(X_new)
        X_new = X_new.T
        x_res = np.empty((X.shape[0], X.shape[1]), dtype=object)
        for p in range(X.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = X_new[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = X[p][j]
        # print(x_res.shape)
        return x_res

    def enc_fit_transform(self, dataset=None, class_name=None):
        """
        given a dataset and the class name, this function applies target encoder on the categorical variables
        self.encdec is the trained encoder
        self.dataset_enc is the encoded dataset

        :param dataset:
        :param class_name:
        :param kwargs:
        :return:
        """
        if self.dataset is None:
            self.dataset = dataset
        if self.class_name is None:
            self.class_name = class_name

        self.features = [c for c in self.dataset.columns if c not in [self.class_name]]
        self.cont_features_names = list(self.dataset[self.features]._get_numeric_data().columns)
        self.cate_features_names = [c for c in self.dataset.columns if c not in self.cont_features_names and c != self.class_name]
        self.cate_features_idx = [self.features.index(f) for f in self.cate_features_names]
        self.cont_features_idx = [self.features.index(f) for f in self.cont_features_names]
        #print(self.cont_features_names, ' cont feature name')
        self.encdec = TargetEncoder(return_df=False)
        dataset_values = self.dataset[self.features].values
        y = self.dataset[self.class_name].values
        self.dataset_enc = self.encdec.fit_transform(dataset_values[:, self.cate_features_idx], y)
        self.dataset_enc_complete = np.zeros((self.dataset_enc.shape[0],len(self.features)))
        #print('index ', self.cate_features_idx, self.cont_features_idx)
        for p in range(self.dataset_enc.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                self.dataset_enc_complete[p][self.cate_features_idx[i]] = self.dataset_enc[p][i]
            for j in self.cont_features_idx:
                self.dataset_enc_complete[p][j] = dataset_values[p][j]
        for i, idx in enumerate(self.cate_features_idx):
            cate_map_i = dict()
            inverse_cate_map_i = dict()
            values = np.unique(dataset_values[:, idx])
            for v1, v2 in zip(dataset_values[:, idx], self.dataset_enc[:, i]):
                cate_map_i[v1] = v2
                inverse_cate_map_i[v2] = v1
                if len(cate_map_i) == len(values):
                    break
            self.cate_map[idx] = cate_map_i
            self.inverse_cate_map.append(inverse_cate_map_i)
        print('cate map ', self.cate_map)
        return self.dataset_enc_complete

    def retrieve_values(self, index, interval, op):
        inverse_dataset = self.dec(self.dataset_enc_complete)
        feature_values = self.dataset_enc_complete[:, index]
        if len(interval) == 1:
            if op == '<':
                indexes = [feature_values.tolist().index(i) for i in feature_values if i <= interval[0]]
            else:
                indexes = [feature_values.tolist().index(i) for i in feature_values if i > interval[0]]
        else:
            index_values_min = [feature_values.tolist().index(i) for i in feature_values if i > interval[0]]
            index_values_max = [feature_values.tolist().index(i) for i in feature_values if i <= interval[1]]
            indexes = list(set(index_values_min) & set(index_values_max))
        res = set(inverse_dataset[indexes,index])
        return list(res)

    #todo fix get cate map
    def get_cate_map(self, i, value):
        found = False
        if i in self.cate_map.keys():
            print('sono in cate map')
            try:
                print(self.cate_map[i][value])
                return self.cate_map[i][value]
            except:
                return value
            '''for key, v in self.cate_map[i].items():
                if v == value:
                    found = True
                    break
            if found == True:
                return key
            else:
                key_list = list(self.cate_map[i].keys())
                val_list = list(self.cate_map[i].values())
                array = np.asarray(val_list)
                idx = (np.abs(array - value)).argmin()
                ind = val_list.index(array[idx])
                return key_list[ind]'''
        else:
            print('questo e un numero')
            return value

    def prepare_dataset(self,df, class_name, numeric_columns, rdf):
        feature_names = df.columns.values
        feature_names = np.delete(feature_names, np.where(feature_names == class_name))
        class_values = np.unique(df[class_name]).tolist()
        numeric_columns = list(df._get_numeric_data().columns)
        real_feature_names = self.__get_real_feature_names(rdf, numeric_columns, class_name)
        features_map = dict()
        for f in range(0, len(real_feature_names)):
            features_map[f] = dict()
            features_map[f][real_feature_names[f]] = np.where(feature_names == real_feature_names[f])[0][0]
        return df, feature_names, features_map, numeric_columns, class_values, rdf, real_feature_names


    def __get_real_feature_names(self,rdf, numeric_columns, class_name):
        if isinstance(class_name, list):
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
        else:
            real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
            real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
        return real_feature_names