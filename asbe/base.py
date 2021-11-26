# AUTOGENERATED! DO NOT EDIT! File to edit: 00_base.ipynb (unless otherwise specified).

__all__ = ['FitTask', 'BaseITEEstimator', 'BaseActiveLearner', 'BaseAcquisitionFunction', 'BaseAssignmentFunction',
           'BaseStoppingRule', 'BaseDataGenerator']

# Cell
#hide_output
from sklearn.base import BaseEstimator
import numpy as np
from typing import Union, Callable, Optional, Tuple, List, Iterator, Any
from copy import deepcopy
from dataclasses import dataclass, field
from sklift.metrics import qini_auc_score, qini_curve
#from pylift.eval import UpliftEval
from fastcore.test import *
import asbe

# Cell
class FitTask(type):
    """Meta class to make preprocessing of data before fitting different models

    This meta class is needed so new models can be incorporated easily
    by only having a .fit() method. It will call a prepare_data function, which
    can also be modified .
    """
    # https://stackoverflow.com/a/18859019
    def __init__(cls, name, bases, clsdict):
        if 'fit' in clsdict:
            def new_fit(self, **kwargs):
                try:
                    X_training = kwargs["X_training"] if "X_training" in kwargs else self.dataset["X_training"]
                    y_training = kwargs["y_training"] if "y_training" in kwargs else self.dataset["y_training"]
                    t_training = kwargs["t_training"] if "t_training" in kwargs else self.dataset["t_training"]
                except:
                    raise ValueError("Can't find (X,t,y) data to fit the model on")
                try:
                    ps_scores  = kwargs["ps_scores"] if "ps_scores" in kwargs else self.dataset["ps_scores"]
                except:
                    ps_scores = None
                try:
                    fun_to_prepare = bases['prepare_data'] if ["prepare_data"] in dir(
                        bases) else self.prepare_data
                    X_training, t_training, y_training, ps_scores = fun_to_prepare(X_training, t_training, y_training, ps_scores)
                except:
                    raise ValueError("Can't prepare data. Use either the default or implement new 'prepare_data' method")
                clsdict['fit'](self, X_training = X_training,
                               t_training = t_training,
                               y_training = y_training,
                               ps_scores = ps_scores)
            setattr(cls, 'fit', new_fit)

# Cell
class BaseITEEstimator(BaseEstimator, metaclass=FitTask):
    """Base class for estimating treatment effects

    This is a base class, that is also usable on its own to estimate treatment effects.
    It works with most models that have a .fit() method, but subclassing is also made
    straightforward throught a metaclass.

    Attributes
    ----------
    model : str, Callable
        treatment effect estimator to be used
    dataset: dict, BaseDataGenerator
        dataset to be used by the model, either offline or online
    two_model: bool
        Switches between S or T learner approach, when appropriate
    ps_model: Callable, None
        Propensity score model, if used, must be classifier

    Methods
    -------
    prepare_data(X_training, t_training, y_training, ps_scores):
        Puts together data for training before .fit() is called

    fit(X_training, t_training, y_training, ps_scores):
        Fits the treatment effect estimator to the training data

    predict(X, ps_scores):
        Predicts the treatment effects based on the supplied X
    """

    def __init__(self,
                model: Union[str, Callable] = None,
                dataset : Union[dict,None] = None,
                two_model: Union[bool,None] = None,
                ps_model : Union[Callable, None] = None,
                **kwargs)-> None:
        self.model = model
        self.dataset = dataset
        self.two_model = two_model
        self.ps_model = ps_model
        self.proba = False
        """
        Makes the estimator ready for its task

        Parameters
        ----------
        model : str, Callable
            treatment effect estimator to be used
        dataset: dict, BaseDataGenerator
            dataset to be used by the model, either offline or online
        two_model: bool
            Switches between S or T learner approach, when appropriate
        ps_model: Callable, None
            Propensity score model, if used, must be classifier

        Returns
        -------
        None
        """

    def _fit_ps_model(self, X_training, t_training):
        self.ps_model.fit(X_training, t_training)

    def prepare_data(self, X_training=None, t_training=None, y_training=None, ps_scores=None):
        """
        Prepares data before fitting the model

        Parameters
        ----------
        X_training : np.ndarray
            Training features (size n x d)
        t_training : np.ndarray
            Training treatments (size n x 1)
        y_training : np.ndarray
            Training labels (size n x 1)
        ps_scores : np.ndarray, None
            Optinal propensity scores if they are coming from outside model

        Returns
        -------
        X_training, t_training, y_training, ps_scores
            Updated training data
        """
        if X_training is None:
            try:
                X_training = deepcopy(self.dataset["X_training"])
                y_training = deepcopy(self.dataset["y_training"])
                t_training = deepcopy(self.dataset["t_training"])
            except:
                raise ValueError("No data to fit the model on.")
        if np.unique(y_training).shape[0] == 2:
            self.proba = True
        if self.ps_model is not None or ps_scores is not None:
            if self.ps_model is not None:
                self._fit_ps_model(X_training, t_training)
                ps_scores = self.ps_model.predict_proba(X_training)
            try:
                X_training = np.hstack((X_training, ps_scores[:,1].reshape((-1, 1))))
            except:
                raise ValueError(f"Shape of propensity scores is {ps_scores.shape},instead of (-1,1)")
        X_t_training = np.hstack((X_training, t_training.reshape(-1,1)))
        if self.two_model is False:
            X_training = deepcopy(X_t_training)

        return X_training, t_training, y_training, ps_scores

    def fit(self,
            X_training,
            t_training,
            y_training,
            ps_scores,
            **kwargs):
        """
        Fits the model using .fit() function of other the given ITE estimators

        Parameters
        ----------
        X_training : np.ndarray
            Training features (size n x d)
        t_training : np.ndarray
            Training treatments (size n x 1)
        y_training : np.ndarray
            Training labels (size n x 1)
        ps_scores : np.ndarray, None
            Optinal propensity scores if they are coming from outside model

        Returns
        -------
        None
        """
        if self.two_model is False or self.two_model is None:
            self.model.fit(X_training, y_training)
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
                    ps_scores = None,
                    **kwargs):
        """
        Predicts the treatment effect using the fitted model on the given features

        Parameters
        ----------
        X : np.ndarray
            Features (size n x d)
        ps_scores : np.ndarray, None
            Optinal propensity scores if they are coming from outside model

        Returns
        -------
        Predicted treatment effect scores (size n x 1 / n x B)
        """
        if self.ps_model is not None or ps_scores is not None:
            if self.ps_model is not None:
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
        return np.asarray(ite)

# Cell
class BaseActiveLearner(BaseEstimator):
    """Basic of Active Learners later used, with the capability for treatment effects

    Inspired by modAL's BaseLearner, however modified for ITE estimation. Dataset is provided by
    a dictionary for easier usage and less clutter, but alternatively a BaseDataGenerator can
    also be supplied.

    Attributes
    ----------
    estimator: BaseEstimator
        Estimator for treatment effects, which expects a BaseEstimator or a subclass of it
    acquisition_function:
        Method that selects units to label
    assignment_function:
        Method that selects control or treatment for selected units
    stopping_function:
        Method that determines if the active learning process should stop after the current step
    dataset : dict, BaseDataGenerator
        Dataset that is used for the active learning process
    al_steps : int = 1
        number of active learnign steps to take. Can be controlled from the outside by self.current_step
    offline : bool
        Determines whether data is supplied beforehand or created on the fly

    Methods
    -------
    fit():
        Fits the supplied estimator with data from dataset
    predict(X):
        Predicts X with the estimator
    query(no_query = None, acquisition_function = None):
        Calls the acquisition function to query datapoints to label.
        Number of queries and AF can be supplied, which overwrited the active learner's
        at the current step
    teach(query_idx=None, assignment_function = None, **kwargs):
        Teaches the supplied indices, by moving them to the training set and
        selects the counterfactuals used
    score(metric="PEHE"):
        Uses a test set to calculate a metric from the estimator
    simulate(no_query: int = None, metric: str = "Qini")
        Simulates the active learning process by going through al_steps.
        The metric for the simulation to be saved can be set
    plot():
        Plots the typical active learning style line plot, after simulation
        has finished.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 acquisition_function: Callable,
                 assignment_function: Union[Callable, list, None],
                 stopping_function: Union[Callable, list, None],
                 dataset: dict,
                 al_steps: int = 1,
                 offline = True,
                 **kwargs
                ) -> None:
        """Initiates the active learning sequence

        If the learning happens offline, sets up calls to the data generating processes

        Parameters
        ----------
        estimator: BaseEstimator
            Estimator for treatment effects, which expects a BaseEstimator or a subclass of it
        acquisition_function:
            Method that selects units to label
        assignment_function:
            Method that selects control or treatment for selected units
        stopping_function:
            Method that determines if the active learning process should stop after the current step
        dataset : dict, BaseDataGenerator
            Dataset that is used for the active learning process
        al_steps : int = 1
            number of active learnign steps to take. Can be controlled from the outside by self.current_step
        offline : bool
            Determines whether data is supplied beforehand or created on the fly

        Returns
        -------
        None
        """
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
        self.next_query = True
        self.offline = offline
        if self.offline is False:
            self.dataset.__dict__["X_training"],\
            self.dataset.__dict__["t_training"],\
            self.dataset.__dict__["y_training"] = self.dataset.get_data(
                self.dataset.no_training)


    def _update_dataset(self, query_idx, **kwargs):
        """Moves data with query_idx indices from pool to training"""
        remove_data = kwargs["remove_data"] if "remove_data" in kwargs else True
        if type(self.dataset) is not dict:
            sds = self.dataset.__dict__
        else:
            sds = self.dataset
        for data in ["X", "t", "y"]:
            if sds[f"{data}_training"].shape[0] > 0 :
                try:
                    sds[f"{data}_training"] = np.concatenate((
                        sds[f"{data}_training"],
                        sds[f"{data}_pool"][query_idx,:]))
                except:
                    sds[f"{data}_training"] = np.concatenate((
                        sds[f"{data}_training"],
                        sds[f"{data}_pool"][query_idx]))
            else:
                sds[f"{data}_training"] = sds[f"{data}_pool"][query_idx,:]
            if remove_data:
                sds[f"{data}_pool"] = np.delete(sds[f"{data}_pool"],
                                                             query_idx, 0)
        try:
            sds["y0_pool"] = np.delete(sds["y0_pool"],
                                                             query_idx, 0)
            sds["y1_pool"] = np.delete(sds["y1_pool"],
                                                             query_idx, 0)
        except:
            pass


    def _select_counterfactuals(self, query_idx, treatment):
        "Rewrites observed outcomes in pool for counterfactuals"
        self.dataset["y_pool"][query_idx] = np.where(treatment == 1,
                                                     self.dataset["y1_pool"][query_idx],
                                                     self.dataset["y0_pool"][query_idx])

        self.dataset["t_pool"][query_idx] = treatment

    def fit(self) -> None:
        """
        Fits the model using the given estimator
        """
        self.estimator.fit(X_training = self.dataset["X_training"],
                           t_training = self.dataset["t_training"],
                           y_training = self.dataset["y_training"])
        return None

    def predict(self, X):
        """
        Predicts using the the given estimator

        Attributes
        ----------
        X : np.ndarray
            The features to predict on

        Returns
        -------
        Predicted treatment effects
        """
        return self.estimator.predict(X=X)

    def query(self, no_query = None, acquisition_function = None):
        """Main function to select datapoints for labeling

        Calls the acquisition function to determine which units to label.
        The acquisition function can be overwritten for a given query and the number
        of queries as well. Queries happen until the budget is exhauested, which is
        controlled by the stopping function.

        Parameters
        ----------
        no_query : int = None
            Number of queries
        acquisition_function
            Optional acquisition function

        Returns
        -------
        X_new, query_idx :
            Selected features, selected indices
        """
        if self.stopping_function is not None:
            self.next_query = self.stopping_function.check_rule(
                self.estimator, self.dataset, self.step)
        if self.next_query:
            if len(self.acquisition_function_list) == 0:
                if acquisition_function is None:
                    acquisition_function = self.acquisition_function
            else:
                if acquisition_function is None:
                    acquisition_function = self.acquisition_function
            if self.offline:
                X_new, query_idx = acquisition_function.select_data(self.estimator,
                                                              self.dataset,
                                                              no_query)
            else:
                X_get = self.dataset.get_X(no_query = no_query)
                data_to_estimate = {"X_training":self.dataset["X_training"],
                                   "X_pool": X_get}
                decision_to_query = acquisition_function.select_data(self.estimator,
                                                              data_to_estimate,
                                                              no_query,
                                                              offline = False)
                if decision_to_query:
                    self.X_to_add = X_get
                    X_new = X_get
                else:
                    X_new = None
                try:
                    query_idx = self.dataset.selected_ix
                except:
                    query_idx = None
        else:
            X_new, query_idx = None, None
        return X_new, query_idx

    def teach(self, query_idx=None, assignment_function = None, **kwargs):
        """Function to assign the selected labels to the training set

        Selects counterfactuals (if possible), through the assignmnet function
        and moves data from the pool to the training set.

        Parameters
        ----------
        query_idx : np.array = None
            the indices of queryed samples
        assignment_function : None
            Optional assignment function to be used tp select counterfactuals

        Returns
        -------
        None
        """

        if len(self.assignment_function_list) == 0:
            if assignment_function is None:
                if self.assignment_function is None:
                    self.assignment_function = asbe.models.RandomAssignmentFunction()
                assignment_function = self.assignment_function
            else:
                raise ValueError("No assignment function provided")
        else:
            if assignment_function is None:
                assignment_function = self.assignment_function
        try:
            treatment = assignment_function.select_treatment(self.estimator,
                                                              self.dataset,
                                                              query_idx)
            if "y0_pool" in self.dataset:
                self._select_counterfactuals(query_idx, treatment)
            matching = treatment == self.dataset["t_pool"][query_idx]
            if matching.all():
                self._update_dataset(query_idx, **kwargs)
            elif matching.sum() > 0:
                self._update_dataset(query_idx[matching], **kwargs)
            else:
                pass
        except:
            treatment_to_add = self.dataset.get_t(X_new = self.X_to_add)
            y = self.dataset.get_y(X_new=self.X_to_add, t_new = treatment_to_add)
            self.dataset.__dict__["X_pool"] = self.X_to_add
            self.dataset.__dict__["t_pool"] = treatment_to_add
            self.dataset.__dict__["y_pool"] = y
            self._update_dataset(np.arange(y.shape[0]), remove_data=True)

        return None

    def score(self, metric="PEHE"):
        """
        Calculates the metric given on the test set.

        Parameters
        ----------
        metric : str = "PEHE"
            Metric to be used when calculating the performance

        Returns
        -------
        The score of the model
        """
        metrics = ["Qini", "PEHE", "Cgains", "decision", "Qini_curve"]
        if not callable(metric):
            if metric not in metrics:
                raise ValueError(f"Please use a valid error ({metrics}), {metric} is not valid")
        try:
            preds = self.estimator.predict(X=self.dataset["X_test"], return_mean=True)
        except:
            preds = self.estimator.predict(X=self.dataset["X_test"])
        if metric == "PEHE":
            try:
                sc = np.sqrt(np.mean(np.square(preds - self.dataset["ite_test"])))
            except KeyError:
                raise Error("Check if dataset contains true ITE values")
        elif metric == "decision":
            dec = np.where((preds >= 0) &( self.dataset["ite_test"] >= 0), 1, 0)
            sc = np.sum(dec)/self.dataset["ite_test"].shape[0]
        elif metric == "Qini":
            sc = qini_auc_score(y_true=self.dataset["y_test"],
                                uplift=preds,
                                treatment=self.dataset["t_test"])
        elif metric == "Qini_curve":
            sc = qini_curve(y_true=self.dataset["y_test"],
                                uplift=preds,
                                treatment=self.dataset["t_test"])
        elif callable(metric):
            try:
                sc = metric(preds, self.dataset["ite_test"])
            except:
                raise ValueError("Metric can't be used with current data")
        return sc

    def simulate(self, no_query: int = None, metric: str = "Qini") -> dict:
        """
        Simulates the active learning process based on the attributes to the class

        Parameters
        ----------
        no_query:
            Number of units to be queried
        Metric:
            Metric to be claculated at the end of each active learning step

        Returns
        -------
        dict
            Dictionary with keys of assignment functions and values of steps and scores
        """
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
            if type(metric) is list:
                for m in metric:
                    res[af.name][m]={}
            for i in range(1, self.al_steps+1):
                self.fit()
                X_new, query_idx = self.query(no_query=no_query, acquisition_function = af)
                if self.next_query:
                    if self.offline:
                        self.teach(query_idx)
                    else:
                        self.teach()
                    if type(metric) is list:
                        for m in metric:
                            res[af.name][m][i] = self.score(metric=m)
                    else:
                        res[af.name][i] = self.score(metric=metric)
                    self.current_step += 1
        self.simulation_results = res
        return res

    def plot(self):
        """Plots the results of the AL simulation"""
        pd.DataFrame(self.simulation_results).plot()

# Cell
class BaseAcquisitionFunction():
    """Base class for acquisition functions

    Based on Eq. 3 in paper:
        - calculate_metrics is used to get infromativeness/representativeness
        - calculate_metrics returns a weighted average between the two
        - select_data does the normalization and selecting top m/Bernoulli draws

    Attributes
    ----------
    no_query : int = 1
        Number of queries
    method : str = "top"
        Way to select the data after metrics are calculated
    name : str
        Name of the AF, used in simulations

    Methods
    -------
    calculate_metrics(model, dataset):
        User defined method to get different metrics about the pool set
    select_data(model, dataset, no_query):
        Based on the calculated metrics, selects the data to be added to the
        training set.
    """
    def __init__(self,
                no_query: int = 1,
                method: str = "top",
                name: str = "base") -> None:
        self.no_query = no_query
        self.method = method
        self.name = name + "_" + str(no_query)

    def calculate_metrics(self, model, dataset) -> np.array:
        """
        Method to calculate base metrics for assignment function

        Parameters
        ----------
        model:
            The ITE estimator
        dataset:
            The dataset dictionary/class that holds the training/pool set.

        Returns
        -------
        weighted average of individual metrics
        """
        return np.arange(dataset["X_pool"].shape[0]) + 1

    def select_data(self, model, dataset, no_query, offline=True):
        """
        Based on the calculated metrics, select data and return X and indexes

        Parameters
        ----------
        model:
            The ITE estimator
        dataset:
            The dataset dictionary/class that holds the training/pool set.
        no_query:
            Number of queries
        offline : bool
            Indicator whether the process is online/offline

        Returns
        -------
        Tuple of new features and indices
        """
        if no_query is None:
            no_query = self.no_query
        if dataset["X_training"].shape[0] = 0:
            try:
                metrics = self.calculate_metrics(model, dataset)
            else:
                metrics = np.random.shuffle(np.arange(dataset["X_pool"].shape[0]))
        else:
            metrics = self.calculate_metrics(model, dataset)

        if offline:
            if self.method == "top":
                query_idx = np.argsort(np.asarray(metrics))[-no_query:][::-1]
            elif self.method == "normalized":
                p_j = (metrics - np.min(metrics))/(np.max(metrics) - np.min(metrics))
                query_idx = []
                for pix in p_j.argsort()[::-1]:
                    if np.random.binomial(1, p = p_j[pix]) == 1:
                        query_idx.append(pix)
                    if len(query_idx) == no_query:
                        break
                query_idx = np.asarray(query_idx)
            else:
                raise NotImplementedError("Please use a method that is implemented")
            X_new = dataset["X_pool"][query_idx,:]
            out = (X_new, query_idx)
        else:
            out = metrics.sum() >= metrics.shape[0]/2
        return out

# Cell
class BaseAssignmentFunction():
    """Base class for assignment functions"""

    def __init__(self, base_selection = 0):
        self.base_selection = base_selection

    def select_treatment(self, model, dataset, query_idx):
        return dataset["t_pool"][query_idx]

# Cell
class BaseStoppingRule():
    """Base class for providing a stopping rule for the active learner"""

    def __init__(self, budget=None):
        self.budget = budget

    def check_rule(self, model, dataset, step):
        if self.budget - step >= 0:
            out = False
        else:
            out = True
        return out

# Cell
@dataclass
class BaseDataGenerator():
    ds: Union[dict, Callable]
    dgp_x : Union[Callable, None] = None
    dgp_t : Union[Callable, None] = None
    dgp_y : Union[Callable, None] = None
    no_training : int = 10

    def __getitem__(self, key):
        return super().__getattribute__(key)

    def get_X(self, method : str = "random", no_query : int = 1):
        if self.ds is not None:
            ix = np.random.randint(low = 0, high = self.ds["X_pool"].shape[0], size=(no_query))
            X_new = self.ds["X_pool"][ix,:]
            self.selected_ix = ix
        else:
            X_new = self.dgp_x(no_query)
        self.X_new = X_new
        return X_new

    def get_t(self, X_new=None):
        if X_new is None:
            X_new = self.X_new
        if self.ds is not None:
            t_new = self.ds["t_pool"][self.selected_ix]
        else:
            t_new = self.dgp_t(X_new)
        self.t_new = t_new
        return t_new

    def get_y(self, X_new=None, t_new=None):
        if X_new is None:
            X_new = self.X_new
        if t_new is None:
            t_new = self.t_new
        if self.ds is not None:
            y_new = np.where(t_new == 1,
                             self.ds["y1_pool"][self.selected_ix],
                             self.ds["y0_pool"][self.selected_ix])
        else:
            y_new = self.dgp_y(X_new, t_new)
        return y_new

    def get_data(self, no_query=1, as_test=False):
        X_new = self.get_X(no_query=no_query)
        t_new = self.get_t()
        y_new = self.get_y()
        if as_test:
            self.__dict__["X_test"] = X_new
            self.__dict__["t_test"] = t_new
            self.__dict__['y_test'] = y_new
        return (X_new, t_new, y_new)