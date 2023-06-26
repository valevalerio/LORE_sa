from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
from abc import ABC, abstractmethod



class Surrogate(ABC):
    """
    Generic surrogate class
    """
    def __init__(self, kind = None, preprocessing=None):
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

    @abstractmethod
    def train(self,prune_tree: dict):
        pass

    def get_model(self):
        pass

class DecTree(Surrogate):

    def __init__(self, kind = None, preprocessing=None):
        super(DecTree, self).__init__(kind, preprocessing)


    def train(self, Z, Yb, weights, class_values, multi_label=False, one_vs_rest=False, cv=5, prune_tree: dict = False):
        dt = DecisionTreeClassifier()
        if prune_tree:
            param_list = {'min_samples_split': [ 0.01, 0.05, 0.1, 0.2, 3, 2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1,  2, 4],
                          'splitter' : ['best', 'random'],
                          'max_depth': [None, 2, 10, 12, 16, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 5, 'auto', 'sqrt', 'log2']
                          }

            if not multi_label or (multi_label and one_vs_rest):
                if len(class_values) == 2 or (multi_label and one_vs_rest):
                    scoring = 'precision'
                else:
                    scoring = 'precision_macro'
            else:
                scoring = 'precision_samples'

            dt_search = sklearn.model_selection.HalvingGridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)
            # print(datetime.datetime.now())
            dt_search.fit(Z, Yb, sample_weight=weights)
            # print(datetime.datetime.now())
            dt = dt_search.best_estimator_
            self.prune_duplicate_leaves(dt)
        else:
            dt.fit(Z, Yb, sample_weight=weights)

        return dt


    def is_leaf(self, inner_tree, index):
        """Check whether node is leaf node"""
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)


    def prune_index(self, inner_tree, decisions, index=0):
        """
        Start pruning from the bottom - if we start from the top, we might miss
        nodes that become leaves during pruning.
        Do not use this directly - use prune_duplicate_leaves instead.
        """
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
            self.is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            # print("Pruned {}".format(index))


    def prune_duplicate_leaves(self, dt):
        """Remove leaves if both"""
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)
