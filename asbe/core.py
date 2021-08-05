# AUTOGENERATED! DO NOT EDIT! File to edit: 01_core.ipynb (unless otherwise specified).

__all__ = ['random_batch_sampling', 'uncertainty_batch_sampling', 'type_s_batch_sampling',
           'expected_model_change_maximization', 'ASLearner', 'estimator_type', 'ITEEstimator', 'variance_based_assf']

# Cell
import numpy as np
from .base import *
from modAL.models.base import BaseLearner
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import Union, Optional, Callable
from copy import deepcopy
from pylift.eval import UpliftEval

# Cell
def random_batch_sampling(classifier, X_pool, n2, **kwargs):
    "Randomly sample a batch from a pool of unlabaled samples"
    n_samples = X_pool.shape[0]
    query_idx = np.random.choice(range(n_samples), size=n2,replace=False)
    return X_pool[query_idx], query_idx

def uncertainty_batch_sampling(classifier, X_pool, n2, **kwargs):
    "Select the top $n_2$ most uncertain units"
    ite_preds, y1_preds, y_preds = classifier.predict(X_pool, **kwargs)
    # Calculate variance based on predicted
    if y1_preds.shape[0] <= 1 or \
    len(y1_preds.shape) <= 1:
            raise Exception("Not possible to calculate uncertainty when dimensions <=1 ")
    ite_vars = np.var(classifier.estimator.y1_preds - classifier.estimator.y0_preds, axis=1)
    query_idx = np.argsort(ite_vars)[-n2:][::-1]

    return X_pool[query_idx], query_idx

def type_s_batch_sampling(classifier, X_pool, n2, **kwargs):
    "Select highest type-s"
    ite_preds, y1_preds, y_preds = classifier.predict(X_pool, **kwargs)
    prob_s = np.sum(ite_preds > 0, axis=1)/ite_preds.shape[1]
    prob_s_sel = np.where(prob_s > 0.5, 1-prob_s, prob_s) + .0001
    query_idx = np.argsort(prob_s_sel)[-n2:][::-1]

    return X_pool[query_idx], query_idx


def expected_model_change_maximization(classifier, X_pool, n2, **kwargs):
    """
    Implementation of EMCM for ITE - using a surrogate SGD model.
    """
    # Get mean of the trained prediction
    ite_train_preds, y1_train_preds, y0_train_preds = \
        classifier.predict(classifier.X_training, **kwargs)
    if ite_train_preds.shape[1] < 1:
        raise ValueError("The treatment effect does not have uncertainty around it - \
                         consider using a different estimator")
    # Get mean of predicted ITE
    ite_pool_preds, y1_pool_preds, y0_pool_preds = \
        classifier.predict(X_pool, **kwargs)
    # Then scale the data so sgd works the best
    sc = StandardScaler()
    X_scaled = sc.fit_transform(classifier.X_training)
    # Fit approx model
    # calc type-s error
    train_type_s_prob_1 = np.sum(ite_train_preds > 0, axis=1)/ite_train_preds.shape[1]
    train_type_s = np.where(train_type_s_prob_1 > 0.5, 1-train_type_s_prob_1, train_type_s_prob_1) + .0001
    pool_type_s_prob_1 = np.sum(ite_pool_preds > 0, axis=1)/ite_pool_preds.shape[1]
    pool_type_s = np.where(pool_type_s_prob_1 > 0.5, 1-pool_type_s_prob_1, pool_type_s_prob_1) + .0001
    classifier.approx_model.fit(
        X = X_scaled,
        y = np.mean(ite_train_preds, axis=1),
        sample_weight = 5*train_type_s)
    # Using list as it is faster than appending to np array
    query_idx = []
    # Using a loop for the combinatorial opt. part
    for ix in range(n2):
        if n2 > (X_pool.shape[0]):
            raise IndexError("Too many samples are queried from the pool ($n_2 > ||X_pool||$)")
        # Select randomly from X_pool
        prob_sampling = np.ones((X_pool.shape[0]))/(X_pool.shape[0]-len(query_idx))
        # Set the probability of already selected samples to 0
        if ix > 0:
            prob_sampling[query_idx] = 0
        # B = 100 by default, can be modified by kwargs
        considered_ixes = np.random.choice(X_pool.shape[0],
                                         size = kwargs["B"] if "B" in kwargs else 100,
                                         replace=False,
                                         p=prob_sampling)
        # Calculate the grads for all the
        grads = np.array([])
        for considered_ix in considered_ixes:
            new_X = sc.transform(X_pool[considered_ix].reshape(1, -1))
            app_predicted_ite = classifier.approx_model.predict(new_X)
            # bootstrapping accroding to eq. 11 of Cai and Zhang
            true_ite = np.random.choice(ite_pool_preds[considered_ix],
                                        size=kwargs["K"] if "K" in kwargs else 5)
            grad = np.sum(np.abs(np.kron((true_ite - app_predicted_ite),new_X)))
            grads = np.append(grads, grad)
        if np.max(grads) < kwargs["threshold"] if "threshold" in kwargs else 0:
            break
        classifier.model_change = np.append(classifier.model_change,np.max(grads))
        query_idx.append(int(considered_ixes[np.argmax(grads)]))
        classifier.approx_model.partial_fit(
            sc.transform(X_pool[int(query_idx[ix])].reshape(1, -1)),
            np.random.choice(ite_pool_preds[int(query_idx[ix])], size=1),
            sample_weight = np.array(pool_type_s[int(query_idx[ix])]).ravel())

    return X_pool[query_idx], query_idx

# Cell
estimator_type = ClassifierMixin
class ASLearner(BaseLearner):
    """A(ctively)S(topping)Learner class for automatic stopping in batch-mode AL"""
    def __init__(self,
                 estimator: estimator_type=None,
                 query_strategy=None,
                 assignment_fc=None,
                 X_training: np.ndarray = None,
                 t_training: np.ndarray = None,
                 y_training: np.ndarray = None,
                 X_pool: np.ndarray = None,
                 X_test: np.ndarray = None,
                 approx_model: RegressorMixin = None
                ) -> None:
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.assignment_fc = assignment_fc
        self.X_training = X_training
        self.y_training = y_training
        self.t_training = t_training
        self.X_pool     = X_pool
        self.X_test     = X_test
        self.approx_model = approx_model
        self.model_change = np.array([])

    def _add_queried_data_class(self, X, t, y):
        self.X_training = np.vstack((self.X_training, X))
        self.t_training = np.concatenate((self.t_training, t))
        self.y_training = np.concatenate((self.y_training, y))

    def _update_estimator_values(self):
        self.estimator.__dict__.update(X_training = self.X_training,
                               y_training  =        self.y_training,
                               t_training  =        self.t_training,
                               X_test      =        self.X_test)

    def teach(self, X_new, t_new, y_new, **kwargs):
        """Teaching new instances to the estimator selected bu the query_strategy

        If no `assignment_fc` is added, all selected samples are used
        If assignment function is added, only those instances are used, where
        $\hat{T} = T$
        """
        if self.assignment_fc is not None:
            X_new, t_new, y_new = self.assignment_fc(
                self, X_new, t_new,
                y_new, simulated=kwargs["simulated"] if "simulated" in kwargs else False)
        else:
            try:
                y_new = np.take_along_axis(y_new, t_new[:, None], axis=1)
            except:
                pass
        self._add_queried_data_class(X_new, t_new.ravel(), y_new.ravel())
        self.fit()

    def fit(self):
        self._update_estimator_values()
        self.estimator.fit()

    def predict(self, X=None, **kwargs):
        """Method for predicting treatment effects within Active Learning

        Default is to predict on the unlabeled pool"""
        if X is None:
            raise Exception("You need to supply an unlabeled pool of instances (with shape (-1,{}))".format(self.X_training.shape[1]))
        self.preds = self.estimator.predict(X, **kwargs)
        return self.preds

    def score(self, preds=None, y_true=None, t_true=None, metric = "Qini"):
        """
        Scoring the predictions - either ITE or observed outcomes are needed.

        If observed outcomes are provided, the accompanying treatments are also needed.
        """
        if metric not in ["Qini", "PEHE", "Cgains"]:
            raise ValueError(f"Please use a valid error (PEHE, Qini, Cgains), {metric} is not valid")
        if metric == "Qini":
            upev = UpliftEval(t_true, y_true, self.preds[0] if preds is None else preds)
            self.scores = upev
            vscore = self.scores.q1_aqini
        if metric == "PEHE":
            vscore = np.sqrt(np.mean(np.square(preds - y_true)))
        if metric == "Cgains":
            upev = UpliftEval(t_true, y_true, self.preds[0] if preds is None else preds)
            self.scores = upev
            vscore = self.scores.cgains
        return vscore

# Cell
class ITEEstimator(BaseEstimator):
    """ Class for building a naive estimator for ITE estimation
    """
    def __init__(self,
                 model: estimator_type = None,
                 two_model: bool = False,
                 ps: Callable = None,
                 **kwargs
                ) -> None:
        self.model = model
        self.two_model = two_model
        self.ps_model = ps

    def _fit_ps_model(self):
        if self.ps_model is not None:
            self.ps_model.fit(self.X_training, self.t_training)

    def fit(self,X_training: np.ndarray = None,
                 t_training: np.ndarray = None,
                 y_training: np.ndarray = None,
                 X_test: np.ndarray = None,
                 ps_scores: np.ndarray = None):

        if X_training is not None:
            self.X_training = X_training
            self.y_training = y_training
            self.t_training = t_training
            self.X_test = X_test
        self.N_training = self.X_training.shape[0]
        try:
            self._fit_ps_model()
            ps_scores = self.ps_model.predict_proba(self.X_training)
        except:
            ps_scores = None
            # if "N_training" not in self.__dict__:
        #     self.N_training = self.X_training.shape[0]
        if ps_scores is not None:
            X_to_train_on = np.hstack((self.X_training, ps_scores[:,1].reshape((-1, 1))))
        else:
            X_to_train_on = self.X_training
        if self.two_model:
            if hasattr(self, "m1") is False:
                self.m1 = deepcopy(self.model)
            control_ix = np.where(self.t_training == 0)[0]
            self.model.fit(X_to_train_on[control_ix,:],
                           self.y_training[control_ix])
            self.m1.fit(X_to_train_on[-control_ix,:],
                        self.y_training[-control_ix])
        else:
            self.model.fit(np.hstack((X_to_train_on,
                                      self.t_training.reshape((self.N_training, -1)))),
                           self.y_training)

    def _predict_without_proba(self, model, X, **kwargs):
        return model.predict(X,
            return_mean = kwargs["return_mean"] if "return_mean" in kwargs else True)

    def _fix_dim_pred(self, preds):
        pred_length = preds.shape[0]
        if preds.shape[1] == 1:
            if np.all(preds == 0):
                preds = np.hstack((preds, np.ones(pred_length).reshape((-1,1))))
            elif np.all(preds == 1):
                preds = np.hstack((preds, np.zeros(pred_length).reshape((-1,1))))
            preds = preds[:, ]
        return preds

    def predict(self, X=None, **kwargs):
        if X is None:
            X = self.X_test
        if self.ps_model is not None and self.ps_model.coef_ is not None:
            pred_ps_scores = self.ps_model.predict_proba(X)[:, 1]
            X = np.hstack((X, pred_ps_scores.reshape(-1, 1)))
        N_test = X.shape[0]
        try:
            if self.two_model:
                self.y1_preds = self.m1.predict_proba(X)
                self.y0_preds = self.model.predict_proba(X)
            else:
                self.y1_preds = self.model.predict_proba(
                                    np.hstack((X,
                                    np.ones(N_test).reshape(-1,1))))
                self.y0_preds = self.model.predict_proba(
                    np.hstack((X,
                               np.zeros(N_test).reshape(-1,1))))
            self.y1_preds = self._fix_dim_pred( self.y1_preds)
            self.y0_preds = self._fix_dim_pred( self.y0_preds)
        except AttributeError:
            try:
                if self.two_model:
                    self.y1_preds = self._predict_without_proba(self.m1, X, **kwargs)
                    self.y0_preds = self._predict_without_proba(self.model, X, **kwargs)
                else:
                    self.y1_preds = self._predict_without_proba(self.model,
                             np.hstack((X,
                             np.ones(N_test).reshape(-1,1))), **kwargs)
                    self.y0_preds = self._predict_without_proba(self.model,
                        np.hstack((X,
                                   np.zeros(N_test).reshape(-1,1))), **kwargs)
            except:
                raise AttributeError("No method found for predicting with the supplied class")
        return self.y1_preds - self.y0_preds, self.y1_preds, self.y0_preds

# Cell
def variance_based_assf(classifier, X, t, y, simulated=False):
    ite_preds, y1_preds, y0_preds = classifier.predict(X, return_mean=False)
    if len(y1_preds.shape) <= 1:
            raise ValueError("Not possible to calculate variance with dim {}".format(y1_preds.shape))
    prop_score = np.var(y1_preds,axis=1)/(
        np.var(y1_preds, axis=1)+np.var(y0_preds,axis=1))
    t_assigned = np.random.binomial(1, prop_score)
    if simulated:
        try:
            y = np.take_along_axis(y, t_assigned[:, None], axis=1)
            t = t_assigned
            usable_units = np.repeat(True, repeats=X.shape[0])
        except:
            raise ValueError("Potential outcomes are needed in a matrix with shape (n,2)")
    else:
        usable_units = np.where(t_assigned == t)
    return X[usable_units], t[usable_units], y[usable_units]