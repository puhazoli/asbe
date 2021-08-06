# AUTOGENERATED! DO NOT EDIT! File to edit: 00_base.ipynb (unless otherwise specified).

__all__ = ['BaseITEEstimator', 'BaseActiveLearner', 'BaseAcquisitionFunction', 'BaseAssignmentFunction']

# Cell
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from typing import Union, Callable, Optional, Tuple, List, Iterator, Any
from copy import deepcopy

# Cell
class BaseITEEstimator(BaseEstimator):
    """Base class for """
    def __init__(self,
                model,
                dataset : Union[dict,None] = None,
                two_model: bool = False,
                ps_model : Union[Callable, None] = None,
                **kwargs)-> None:
        self.model = model
        self.dataset = dataset
        self.two_model = two_model
        self.ps_model = ps_model
        self.proba = False

    def _fit_ps_model(self, X_training, t_training):
        self.ps_model.fit(X_training, t_training)

    def fit(self,
            X_training: Union[np.ndarray, None] = None,
            t_training: Union[np.ndarray, None] = None,
            y_training: Union[np.ndarray, None] = None,
            ps_scores: Union[np.ndarray, None] = None):
        "Fit function, using .fit() from other models"
        if X_training is None:
            X_training = deepcopy(self.X_training)
            y_training = deepcopy(self.y_training)
            t_training = deepcopy(self.t_training)

        if np.unique(y_training).shape[0] == 2:
            self.proba = True
        if not self.ps_model and ps_scores is not None:
            self._fit_ps_model(X_training, t_training)
            ps_scores = self.ps_model.predict(X_training)
            try:
                X_training = np.hstack((X_training, ps_scores[:,1].reshape((-1, 1))))
            except:
                raise ValueError(f"Shape of propensiry scores is {ps_scores.shape},instead of (-1,1)")

        X_t_training = np.hstack((X_training, t_training.reshape(-1,1)))
        if self.two_model is False or self.two_model is None:
            self.model.fit(X_t_training, y_training)
        else:
            control_ix = np.where(t_training == 0)[0]
            self.m1 = deepcopy(self.model)
            self.model.fit(X_training[control_ix,:], y_training[control_ix])
            self.m1.fit(X_training[-control_ix,:],   y_training[-control_ix])

    def _predict_bin_or_con(self, model, X):
        if self.proba:
            try:
                out = model.predict_proba(X)[:, 1]
            except:
                out = model.predict(X)
        else:
            out = model.predict(X)
        return out

    def predict(self,
                    X: np.ndarray,
                    ps_scores = None):
        if self.ps_model is not None and ps_scores is None:
            ps_scores = self.ps_model.predict_proba(X)
            X = np.hstack((X, ps_scores[:,1].reshape((-1, 1))))
        if self.two_model:
            m0_preds = self._predict_bin_or_con(self.model, X)
            m1_preds =  self._predict_bin_or_con(self.m1, X)
            ite = m1_preds - m0_preds
        else:
            try:
                X0 = np.hstack((X, np.zeros((X.shape[0], 1))))
                X1 = np.hstack((X, np.ones((X.shape[0], 1))))
                m1_preds = self._predict_bin_or_con(self.model, X1)
                m0_preds = self._predict_bin_or_con(self.model, X0)
                ite = m1_preds - m0_preds
            except:
                ite = self._predict_bin_or_con(self.model, X)
        return np.array(ite)

# Cell
class BaseActiveLearner(BaseEstimator):
    """Basic of Active Learners later used, with the capability for treatment effects

    Inspired by modAL's BaseLearner, however modified for ITE estimation. Dataset is provided by
    a dictionary for easier usage and less clutter.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 acquisition_function: Callable,
                 assignment_function: Union[Callable, list, None],
                 stopping_function: Union[Callable, list, None],
                 dataset: dict,
                 al_steps: int = 1,
                 **kwargs
                ) -> None:
        self.estimator = estimator
        if type(acquisition_function) is list:
            self.acquisition_function_list = acquisition_function
        else:
            self.acquisition_function = acquisition_function
            self.acquisition_function_list = []
        if type(assignment_function) is list:
            self.assignment_function_list = assignment_function
        else:
            self.assignment_function = assignment_function
            self.assignment_function_list = []
        self.stopping_function = stopping_function
        self.dataset = deepcopy(dataset)
        self.al_steps = al_steps
        self.current_step = 0

    def _update_dataset(self, query_idx, treatment, **kwargs):
        """Moves data with query_idx indices from pool to training"""
        remove_data = kwargs["remove_data"] if "remove_data" in kwargs else True
        for data in ["X", "t", "y"]:
            if self.dataset[f"{data}_training"].shape[0] > 0 :
                try:
                    self.dataset[f"{data}_training"] = np.concatenate((
                        self.dataset[f"{data}_training"],
                        self.dataset[f"{data}_pool"][query_idx,:]))
                except:
                    self.dataset[f"{data}_training"] = np.concatenate((
                        self.dataset[f"{data}_training"],
                        self.dataset[f"{data}_pool"][query_idx]))
            else:
                self.dataset[f"{data}_training"] = self.dataset[f"{data}_pool"][query_idx,:]
            if remove_data:
                self.dataset[f"{data}_pool"] = np.delete(self.dataset[f"{data}_pool"],
                                                             query_idx, 0)

    def fit(self) -> None:
        self.estimator.fit(self.dataset["X_training"],
                           self.dataset["t_training"],
                           self.dataset["y_training"])
        return None

    def predict(self, X):
        self.estimator.predict(X)

    def query(self, no_query = None, acquisition_function = None):
        """Main function to get labels of datapoints"""
        if len(self.acquisition_function_list) == 0:
            if acquisition_function is None:
                acquisition_function = self.acquisition_function
        else:
            if acquisition_function is None:
                acquisition_function = self.acquisition_function
        query_idx = acquisition_function.select_data(self.estimator,
                                                          self.dataset,
                                                          no_query)
        return query_idx

    def teach(self, query_idx, assignment_function = None, **kwargs):
        if len(self.assignment_function_list) == 0:
            if assignment_function is None:
                assignment_function = self.assignment_function
            else:
                raise ValueError("No assignment function provided")
        else:
            if assignment_function is None:
                assignment_function = self.assignment_function
        treatment = assignment_function.select_treatment(self.estimator,
                                                              self.dataset,
                                                              query_idx)
        matching = treatment == self.dataset["t_pool"][query_idx]

        if matching.all():
            self._update_dataset(query_idx, treatment, **kwargs)
        elif matching.sum() > 0:
            self._update_dataset(query_idx[matching], treatment[matching], **kwargs)
        else:
            pass

    def score(self, metric="PEHE"):
        if metric not in ["Qini", "PEHE", "Cgains"]:
            raise ValueError(f"Please use a valid error (PEHE, Qini, Cgains), {metric} is not valid")
        if metric == "PEHE":
            preds = self.estimator.predict(self.dataset["X_test"])
            try:
                sc = np.sqrt(np.mean(np.square(preds - self.dataset["ite_test"])))
            except KeyError:
                raise Error("Check if dataset contains true ITE values")
        return sc

    def simulate(self, no_query: int = None) -> dict:
        ds = deepcopy(self.dataset)
        est = deepcopy(self.estimator)
        res = {}
        if len(self.acquisition_function_list) == 0:
            laf = [self.acquisition_function]
        else:
            laf = self.acquisition_function_list
        for af in laf:
            if no_query is None:
                no_query = af.no_query
            self.dataset = deepcopy(ds)
            self.estimator = deepcopy(est)
            self.current_step = 0
            res[af.name] = {}
            for i in range(1, self.al_steps+1):
                self.fit()
                X_new, query_idx = self.query(no_query=no_query, acquisition_function = af)
                self.teach(query_idx)
                preds = self.predict(self.dataset["X_test"])
                res[af.name][i] = self.score()
                self.current_step += 1
        return res

# Cell
class BaseAcquisitionFunction():
    """Base class for acquisition functions"""
    def __init__(self,
                no_query: int = 1,
                method: str = "top",
                name: str = "base") -> None:
        self.no_query = no_query
        self.method = method
        self.name = name + "_" + str(no_query)

    def calculate_metrics(self, model, dataset) -> np.array:
        #return model.predict(dataset["X_pool"])
        return np.arange(dataset["X_pool"].shape[0])

    def select_data(self, model, dataset, no_query):
        if no_query is None:
            no_query = self.no_query
        metrics = self.calculate_metrics(model, dataset)
        if self.method == "top":
            query_idx = np.argsort(metrics)[-no_query:][::-1]
            X_new = dataset["X_pool"][query_idx,:]
        return X_new, query_idx

# Cell
class BaseAssignmentFunction():
    """Base class for assignment functions"""
    def __init__(self):
        pass

    def select_treatment(self, model, dataset, query_idx):
        return dataset["t_pool"][query_idx]