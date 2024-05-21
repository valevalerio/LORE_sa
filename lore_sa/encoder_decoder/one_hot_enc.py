import itertools

from sklearn.compose import ColumnTransformer

from .enc_dec import EncDec
import numpy as np
import copy

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, OrdinalEncoder

__all__ = ["EncDec", "OneHotEnc"]


class InvertableColumnTransformer(ColumnTransformer):
    """
    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.

    Warning this is flaky and use at your own risk.  Validation checks that the column count in
    `transformers` are in your object `X` to be inverted.  Reordering of columns will break things!

    taken from: https://github.com/scikit-learn/scikit-learn/issues/11463
    """
    def inverse_transform(self, X):
        print(X)
        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            print(name, indices.start, indices.stop, transformer)
            arr = X[:, indices.start: indices.stop]
            if transformer in (None, "passthrough", "drop"):
                pass
            else:
                arr = transformer.inverse_transform(arr)
            arrays.append(arr)
        retarr = np.concatenate(arrays, axis=1)
        reordered_array = np.zeros_like(retarr)

        # apply the original order of the columns
        dest_indexes = []
        for t in self.transformers_:
            dest_indexes.extend(t[2])

        for i, d in enumerate(dest_indexes):
            reordered_array[:, i] = retarr[:, d]

        return reordered_array


class OneHotEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions. 
    It relies on OneHotEncoder class from sklearn
    """

    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type='one-hot'
        self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)

        # from the dataset descriptor, we extract the number of features, calculating the maximum
        # index of the features in the sub-fields of the descriptor: categorical, numeric, target
        max_index = 0
        for l in ['numeric', 'categorical', 'ordinal', 'target']:
            if l in self.dataset_descriptor:
                for k, v in self.dataset_descriptor[l].items():
                    if v['index'] > max_index:
                        max_index = v['index']
        categories = np.zeros(max_index + 1, dtype=object)

        for l in ['numeric', 'categorical', 'ordinal', 'target']:
            if l in self.dataset_descriptor:
                if l == 'numeric':
                    for k, v in self.dataset_descriptor[l].items():
                        categories[v['index']] = [v['min'], v['max']]
                else:
                    for k, v in self.dataset_descriptor[l].items():
                        categories[v['index']] = v['distinct_values']
        # create a dataset to fit the encoder with the categories
        # each entry of the datasets should contain the corersponding values in the right position
        # of the categories array
        mock_data = list(itertools.product(*categories))

        # Create the column transformer to apply OneHotEncoder only to categorical features
        self.encoder = InvertableColumnTransformer(
            transformers=[
                ('numeric', FunctionTransformer(lambda x: x), [ v['index'] for v in self.dataset_descriptor['numeric'].values()]),
                ('categorical', OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=np.int16), [ v['index'] for v in self.dataset_descriptor['categorical'].values()]),
                ('target', OrdinalEncoder(dtype=np.int16), [ v['index'] for v in self.dataset_descriptor['target'].values()])
            ],
            remainder='passthrough'
        )

        self.encoder.fit(mock_data)

        print('transformers', self.encoder.transformers_)
        print('output indices', self.encoder.output_indices_)
        print('named transformers', self.encoder.named_transformers_.get('categorical').categories_)

        for name, indices in self.encoder.output_indices_.items():
            print(name, indices.start, indices.stop)
            if name == 'categorical':
                print(self.encoder.named_transformers_.get(name).categories_)
                cat_categories = self.encoder.named_transformers_.get(name).categories_
                i = indices.start
                for j, k in enumerate(self.dataset_descriptor['categorical'].keys()):
                    print('categorical', k,  cat_categories[j], self.dataset_descriptor['categorical'][k]['index'], i)
                    for v in cat_categories[j]:
                        print('   ', i, f"{k}={v}")
                        i += 1
            if name == 'target':
                print(self.encoder.named_transformers_.get(name).categories_)
                target_categories = self.encoder.named_transformers_.get(name).categories_
                i = indices.start
                for j, k in enumerate(self.dataset_descriptor['target'].keys()):
                    print('target', k,  target_categories[j], self.dataset_descriptor['target'][k]['index'], i)
                    for v in target_categories[j]:
                        print('   ', i, f"{k}={v} [{self.encoder.named_transformers_.get(name).transform([[v]])[0]}]")
                    i += 1
            if name == 'numeric':
                print('numeric', indices.start, indices.stop)
                i = indices.start
                for k, v in self.dataset_descriptor['numeric'].items():
                    print('   ',i , k)
                    i += 1


        if self.dataset_descriptor.get("categorical") is None:
            raise Exception("Dataset descriptor is malformed for One-Hot Encoder: 'categorical' key is not present")

    def encode(self, X: np.array):
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """

        print('encoding using Scikit one hot encoder')
        print('x', X)
        encoded = self.encoder.transform(X)
        print('encoded scikit', encoded)
        print('dtype', encoded.dtype)

        # encoded_feature_list = []
        # original_encoded_feature_list = []
        # # self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)
        # for k in self.encoded_descriptor['categorical'].keys():
        #
        #     label_dict = self.encoded_descriptor['categorical'][k]
        #     label_index = label_dict['index']
        #
        #     mapping = {}
        #     for value in range(len(label_dict['distinct_values'])):
        #         mapping[label_dict['distinct_values'][value]] = value
        #
        #     arr = list(np.zeros(len(label_dict['distinct_values']), dtype=int))
        #     arr[mapping[x[label_index]]] = 1
        #     x = np.delete(x, label_index)
        #     x = np.insert(x, label_index, arr)
        #
        #     encoded_feature = {(label_index+i):"=".join([k, v])  for i,v in enumerate(label_dict['distinct_values'])}
        #     self.encoded_features.update(encoded_feature)
        #     encoded_feature_list.append(encoded_feature)
        #     original_encoded_feature_list.append(str(k))
        #     self.update_encoded_index(str(k),len(label_dict['distinct_values'])-1)
        #
        # self.clean_encoded_descriptor_by_old(original_encoded_feature_list)
        # self.add_encoded_features(encoded_feature_list)
        # print('encoded feature scikit', self.encoder.get_feature_names_out())


        return encoded

    def update_encoded_index(self,current_field, size: int):
        current_index_value = self.encoded_descriptor['categorical'][current_field]['index']
        for type in self.encoded_descriptor.keys():
            for k in self.encoded_descriptor[type]:
                if k != current_field:
                    original_index = self.encoded_descriptor[type][k]['index']
                    if original_index>current_index_value:
                        self.encoded_descriptor[type][k]['index'] = self.encoded_descriptor[type][k]['index'] + size

    def clean_encoded_descriptor_by_old(self,old_field):
        for current_field in old_field:
            #remove old field
            self.encoded_descriptor['categorical'].pop(current_field)

    def add_encoded_features(self, encoded_features):
        for feature in encoded_features:
            #add new features encoded
            new_encoded_feature = {v:dict(index=k) for k,v in feature.items()}
            self.encoded_descriptor['categorical'].update(new_encoded_feature)


    def get_encoded_features(self):
        if self.encoded_features is None:
            raise Exception("You have not run the encoder yet")
        else:
            for type in self.encoded_descriptor.keys():
                if type == "categorical":
                    continue
                else:
                    for k in self.encoded_descriptor[type]:
                        self.encoded_features.update({self.encoded_descriptor[type][k]['index']:k})

            return dict(sorted(self.encoded_features.items()))

    def __str__(self):
        if len(self.encoded_features) > 0:
            return "OneHotEncoder - features encoded: %s" % (",".join(self.encoded_features.values()))
        else:
            return "OneHotEncoder - no features encoded"

    def decode(self, Z: np.array):
        """
        Decode the array staring from the original descriptor

        :param [Numpy array] x: Array to decode
        :return [Numpy array]: Decoded array
        """
        decoded = self.encoder.inverse_transform(Z)
        print('decoded inverted transformer', decoded)
        print('encoded feature scikit', self.encoder.named_transformers_.get('cat').categories_)

        return decoded

        # print('dataset descriptor keys', self.dataset_descriptor.keys())
        # print('dataset descriptor numeric', self.dataset_descriptor['numeric'])
        # for l in ['numeric', 'categorical']:
        #     for k, v in self.dataset_descriptor[l].items():
        #         print(v['index'], k, v, l)
        #
        # for l in ['numeric', 'categorical']:
        #     for k, v in self.encoded_descriptor[l].items():
        #         print(v['index'], k, v, l)
        #
        # for k in self.dataset_descriptor['categorical'].keys():
        #     label_dict = self.dataset_descriptor['categorical'][k]
        #     label_index = label_dict['index']
        #
        #     mapping = {}
        #     for value in range(len(label_dict['distinct_values'])):
        #         mapping[label_dict['distinct_values'][value]] = value
        #
        #
        #     for l in range(len(label_dict['distinct_values'])):
        #         arr = list(np.zeros(len(label_dict['distinct_values']), dtype=int))
        #         arr[l] = 1
        #         mapping[list(mapping.keys())[l]] = [int(x) for x in arr]
        #
        #     code = [int(x) for x in x[label_index: label_index + len(label_dict['distinct_values'])]]
        #     for t in mapping.keys():
        #         if list(mapping[t]) == code:
        #             label = t
        #
        #     x = np.concatenate((x[:label_index], [label], x[label_index+len(label_dict['distinct_values']):]),axis=0)
        #
        # return x
