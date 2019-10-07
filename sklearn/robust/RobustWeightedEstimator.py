# coding: utf-8

# Author: Timothee Mathieu
#
# License: BSD 3 clause

import numpy as np
import warnings

from ..base import BaseEstimator, MetaEstimatorMixin, clone
from ..linear_model import SGDClassifier, SGDRegressor

from sklearn.robust.mean_estimators import MOM, blockMOM, huber

def _huber_psisx(x,c):
    out = x>c
    return out + (1-out)*(2*(x>0)-1)*c/x

def _mom_psisx(med_block,n):
    res = np.zeros(n)
    res[med_block] = 1
    return lambda x : res


class RobustWeightedEstimator(MetaEstimatorMixin, BaseEstimator):
    """
    Doc: TODO
    """
    def __init__(self, base_estimator=None, weighting="huber", max_iter=100,
                  c=1.35, K=3, loss=None):
        self.base_estimator=base_estimator
        self.weighting=weighting
        self.c=c
        self.K=K
        self.loss=loss
        self.max_iter=max_iter

    def fit(self,X,y):
        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = SGDRegressor()
        base_estimator.warm_start=True
        base_estimator.max_iter=len(X)
        loss=self._get_loss_function()
        base_estimator.partial_fit(X,y)

        for epoch in range(self.max_iter):
            loss_values=loss(base_estimator.predict(X),y)
            weights=self._weighting(loss_values)
            base_estimator.partial_fit(X, y, sample_weight=weights)

        self.weights = weights
        self.estimator = base_estimator

    def _get_loss_function(self):
        if self.loss is None:
            warnings.warn("RobustWeightedEstimator: No loss"
                          " function given. Using square loss"
                          " function for regression.")
            return lambda x,y :(x-y)**2
        else:
            return self.loss

    def _weighting(self,loss_values):
        if self.weighting == 'huber':
            psisx = lambda x : _huber_psisx(x,self.c)
            mu=huber(loss_values,self.c)

        elif self.weighting == 'mom':
            blocks=blockMOM(loss_values,self.K)
            mu,idmom=MOM(loss_values,blocks)
            psisx= _mom_psisx(blocks[idmom],len(loss_values))
        else :
            raise ValueError("RobustWeightedEstimator: "
                             "no such weighting scheme")
        w=psisx(loss_values-mu)
        return w/np.sum(w)*len(loss_values)

    def predict(self,X):
        return self.estimator.predict(X)
