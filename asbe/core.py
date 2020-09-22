# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['random_batch_sampling', 'uncertainty_batch_sampling', 'expected_model_change_maximization',
           'variance_assignment', 'ASLearner', 'estimator_type', 'ITEEstimator']

# Cell
import numpy as np

from modAL.models.base import BaseLearner
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from typing import Union, Optional
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

def expected_model_change_maximization(classifier, X_pool, n2, **kwargs):
    """
    Implementation of EMCM for ITE - using a surrogate SGD model.
    """
    # Get mean of the trained prediction
    ite_train_preds, y1_train_preds, y0_train_preds = \
        classifier.predict(classifier.X_training, **kwargs)
    if ite_train_preds.shape[1] < 1:
        raise ValueError("The treatment effect does not uncertainty around it - \
                         consider using a different estimator")
    # Get mean of predicted ITE
    ite_pool_preds, y1_pool_preds, y0_pool_preds = \
        classifier.predict(X_pool, **kwargs)
    # Then scale the data so sgd works the best
    sc = StandardScaler()
    X_scaled = sc.fit_transform(classifier.X_training)
    # Fit approx model
    classifier.approx_model.fit(
        X_scaled,
        ite_train_preds if ite_train_preds.shape[1] <= 1 else np.mean(ite_train_preds, axis=1))
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
            np.random.choice(ite_pool_preds[int(query_idx[ix])], size=1))

    return X_pool[query_idx], query_idx

# Cell
def variance_assignment(classifier, X_pool, n2, **kwargs):
    """Function to assign treatment or control based on the variance of the cf
    """
    ite_pool_preds, y1_pool_preds, y0_pool_preds = \
        classifier.predict(X_pool, **kwargs)
    var_y1 = np.var(y1_pool_preds, axis=1)
    var_y0 = np.var(y0_pool_preds, axis=1)
    prob_of_treatment = var_y1/(var_y1+var_y0)
    drawn_treatment = np.random.binomial(1, prob_of_treatment)
    return drawn_treatment

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

    def teach(self, X_new, t_new, y_new):
        """Teaching new instances to the estimator selected bu the query_strategy

        If no `assignment_fc` is added, all selected samples are used
        If assignment function is added, only those instances are used, where
        $\hat{T} = T$
        """
        if self.assignment_fc is None:
            self._add_queried_data_class(X_new, t_new, y_new)
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
        if metric == "Qini":
            upev = UpliftEval(t_true, y_true, self.preds[0] if preds is None else preds)
            self.scores = upev
        return self.scores.q1_aqini

# Cell
class ITEEstimator(BaseEstimator):
    """ Class for building a naive estimator for ITE estimation
    """
    def __init__(self,
                 model: estimator_type = None,
                 two_model: bool = False,
                 **kwargs
                ) -> None:
        self.model = model
        self.two_model = two_model

    def fit(self,X_training: np.ndarray = None,
                 t_training: np.ndarray = None,
                 y_training: np.ndarray = None,
                 X_test: np.ndarray = None):
        if X_training is not None:
            self.X_training = X_training
            self.y_training = y_training
            self.t_training = t_training
            self.X_test = X_test
        self.N_training = self.X_training.shape[0]
        # if "N_training" not in self.__dict__:
        #     self.N_training = self.X_training.shape[0]
        if self.two_model:
            if self.m1 is None:
                self.m1 = deepcopy(self.model)
            control_ix = np.where(self.t_training == 0)[0]
            self.model.fit(self.X_training[control_ix,:],
                           self.y_training[control_ix])
            self.m1.fit(self.X_training[-control_ix,:],
                        self.y_training[-control_ix])
        else:
            self.model.fit(np.hstack((self.X_training,
                                      self.t_training.reshape((self.N_training, -1)))),
                           self.y_training)

    def _predict_without_proba(self, model, X, **kwargs):
        return model.predict(X,
            return_mean = kwargs["return_mean"] if "return_mean" in kwargs else True)

    def predict(self, X=None, **kwargs):
        if X is None:
            X = self.X_test
        N_test = X.shape[0]
        print(self.m1.predict_proba(X))
        try:
            if self.two_model:
                self.y1_preds = self.m1.predict_proba(X)[:,1]
                self.y0_preds = self.model.predict_proba(X)[:,1]
            else:

                self.y1_preds = self.model.predict_proba(
                                    np.hstack((X,
                                    np.ones(N_test).reshape(-1,1))))[:,1]
                self.y0_preds = self.model.predict_proba(
                    np.hstack((X,
                               np.zeros(N_test).reshape(-1,1))))[:,1]
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