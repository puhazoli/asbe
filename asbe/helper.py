# AUTOGENERATED! DO NOT EDIT! File to edit: 99_misc.ipynb (unless otherwise specified).

__all__ = ['get_ihdp_dict']

# Cell
from .base import *
from .models import *
from .estimators import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Cell
def get_ihdp_dict(i:int = 1):
    df = pd.read_csv(
    f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{i}.csv",
    names = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f'x{x}' for x in range(25)])

    X = df.loc[:,"x0":].to_numpy()
    t = df["treatment"].to_numpy()
    #t = np.zeros_like(t)
    y = df["y_factual"].to_numpy()
    y1 = np.where(df["treatment"] == 1,
                   df['y_factual'],
                   df['y_cfactual'])
    y0 = np.where(df["treatment"] == 0,
                   df['y_factual'],
                   df['y_cfactual'])
    ite = np.where(df["treatment"] == 1,
                   df['y_factual'] - df["y_cfactual"],
                   df['y_cfactual'] - df["y_factual"])
    X_train, X_test, t_train, t_test, y_train, y_test, ite_train, ite_test, y1_train, y1_test, y0_train, y0_test = train_test_split(
    X, t, y, ite, y1, y0,  test_size=0.9, random_state=1005)
    ds = {"X_training": X_train,
     "y_training": y_train,
     "t_training": t_train,
     "X_pool": deepcopy(X_test),
     "y_pool": deepcopy(y_test),
     "t_pool": deepcopy(t_test),
     "y1_pool": y1_test,
     "y0_pool":y0_test,
     "X_test": X_test,
     "y_test": y_test,
      "t_test": t_test,
      "ite_test": ite_test
     }
    return ds