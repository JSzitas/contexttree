from typing import Self

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from . import network

class ContextTreeBase(object):
    """Abstract base method to inherit from - so we get to reuse 
    code between Classifier and Regressor models
    """
    def __init__(self,
                 n_tree: int = 50,
                 max_depth: int = 4,
                 nn_rl_rounds: int = 500,
                 nn_layer_dims: [int] = [120,120,120],
                 nn_dropout_rate: float = 0.2):
        """Common initializer for both the Regressor and Classifier models"""
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.nn_rounds = nn_rl_rounds
        self.nn_layer_dims = nn_layer_dims
        self.nn_dropout_rate = nn_dropout_rate
        self.forest = None
        self.nn = None
    # overloading init later and implementing everything as abstract methods enables us
    # to only implement fit once, here: 
    def fit(self, X: np.array, y: np.array, train_frac: float= 0.7) -> Self:

        # add standardizer here


        # fit forest 
        self.forest = self.forest.fit(X, y)
        # initialize contextual network   
        self.nn = network.ContextualNetNetwork(X.cols, self.nn_layer_dims,
                                               self.nn_dropout_rate, self.n_tree)
        # run RL loop to train network 
        #for i in range(self.nn_rounds):
            # sample actions for a given random batch 
            #batch = 
            # forward pass
            #preds = self.nn(batch)
        # overload is only on the reward

        return self
    def predict(self, X: np.array) -> None:
        return None
    def reward(self) -> None:
        return None


class ContextTreeClassifier(ContextTreeBase):
    """Contextual Tree Classifier - wraps Random Forest and Contextual Network"""

    def __init__(self,
                 n_tree: int = 50,
                 max_depth: int = 4,
                 nn_rl_rounds: int = 500,
                 nn_layer_dims: [int] = [120,120,120],
                 nn_dropout_rate: float = 0.2):
        # inherit initializer from parent class and extend by adding forest type
        super().__init__(n_tree, max_depth,
         nn_rl_rounds, nn_layer_dims, nn_dropout_rate)
        self.forest = RandomForestClassifier(
            n_estimators=self.n_tree, criterion='entropy')

    def reward(self, X: np.array, y: np.array, tree_id: int) -> [float]:
        # get predictions for a given tree
        pred = self.forest.estimators_[tree_id].predict(X)
        # get rewards for RL algorithm as in original paper - 
        # 1 if the class mathes, and -1 otherwise
        rewards = y - pred
        # set the cases where they are not equal to -1, otherwise set to 1
        rewards = [-1.0 if reward != 0.0 else 1.0 for reward in rewards]
        return rewards

class ContextTreeRegressor(ContextTreeBase):
    """Contextual Tree Regressor - wraps Random Forest and Contextual Network"""

    def __init__(self,
                 n_tree: int = 50,
                 max_depth: int = 4,
                 nn_rl_rounds: int = 500,
                 nn_layer_dims: [int] = [120,120,120],
                 nn_dropout_rate: float = 0.2):
        # inherit initializer from parent class and extend by adding forest type
        super().__init__(n_tree, max_depth,
         nn_rl_rounds, nn_layer_dims, nn_dropout_rate)
        self.forest = RandomForestRegressor(
            n_estimators=self.n_tree, criterion='entropy')

    def reward(self, X: np.array, y: np.array, tree_id: int) -> [float]:
        # get predictions for all trees
        preds = [tree.predict(X) for tree in self.forest.estimators_]
        # now convert to squared errors 
        preds = [(y-pred)**2 for pred in preds]
        # now convert that using the 1-2d reward function, where d is the 
        # result of normalizing losses between 0 and 1
         #normalizations = []


        pass
        #return rewards




