import datetime

from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
from abc import ABC, abstractmethod

from lore_sa.logger import logger

__all__ = ["DecTree"]

class Surrogate(ABC):
    """
    Generic surrogate class
    """
    def __init__(self, kind = None, preprocessing =None):
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

    @abstractmethod
    def train(self, Z, Yb, weights):
        pass

    @abstractmethod
    def get_model(self):
        pass

class DecTree(Surrogate):
    """
    ..
    """

    def __init__(self, kind = None, preprocessing=None):
        super().__init__(kind, preprocessing)
        self.dt = None

    def train(self, Z, Yb, weights = None, class_values = None, multi_label: bool= False, one_vs_rest: bool = False, cv = 5, prune_tree: bool = False):
        """

        :param Z: The training input samples
        :param Yb: The target values (class labels) as integers or strings.
        :param weights: Sample weights.
        :param class_values:
        :param [bool] multi_label:
        :param [bool] one_vs_rest:
        :param [int] cv:
        :param [bool] prune_tree:
        :return:
        """
        self.dt = DecisionTreeClassifier()
        if prune_tree is True:
            param_list = {'min_samples_split': [ 0.01, 0.05, 0.1, 0.2, 3, 2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1,  2, 4],
                          'splitter' : ['best', 'random'],
                          'max_depth': [None, 2, 10, 12, 16, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 5, 'auto', 'sqrt', 'log2']
                          }

            if multi_label is False or (multi_label is True and one_vs_rest is True):
                if len(class_values) == 2 or (multi_label and one_vs_rest):
                    scoring = 'precision'
                else:
                    scoring = 'precision_macro'
            else:
                scoring = 'precision_samples'

            dt_search = sklearn.model_selection.HalvingGridSearchCV(self.dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)
            logger.info('Search the best estimator')
            logger.info('Start time: {0}'.format(datetime.datetime.now()))
            dt_search.fit(Z, Yb, sample_weight=weights)
            logger.info('End time: {0}'.format(datetime.datetime.now()))
            self.dt = dt_search.best_estimator_
            logger.info('Pruning')
            self.prune_duplicate_leaves(self.dt)
        else:
            self.dt.fit(Z, Yb, sample_weight=weights)

        return self.dt


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
            logger.info("Pruned {}".format(index))


    def prune_duplicate_leaves(self, dt):
        """Remove leaves if both"""
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)

    def get_model(self):
        return self.dt
