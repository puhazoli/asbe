# Automatic Stopping for Batch-mode Experimentation
> Code for using active learning and automatic stopping for designing experiments


Created with nbdev by ZP

## Install

`python -m pip install  git+https://github.com/puhazoli/asbe`

## How to use
ASBE builds on the functional views of modAL, where an AL algorithm can be run by putting together pieces. You need the following ingredients:
- an ITE estimator (`ITEEstimator()`),
- an acquisition function,
- and an assignment function.
- Additionaly, you can add a stopping criteria to your model. 
If all the above are defined, you can construct an `ASLearner`, which will help you in the active learning process.

```python
from asbe.core import *
from sklearn.linear_model import LogisticRegression
import numpy as np
```

```python
N = 1000
X = np.random.normal(size = N).reshape((N/2,2))
t = np.random.binomial(n = 1, p = 0.5, size = N/2)
y = np.random.binomial(n = 1, p = 1/(1+np.exp(X[:, 1]*2 + t*3)))
a = ITEEstimator(LogisticRegression(solver="lbfgs"))
a.fit(X, t, y)
```

## Learning actively
Similarly, you can create an `ASLearner`, for which you will initialize the dataset and set the preferred modeling options. Let's see how it works:
- we will use XBART to model the treatment effect with a one-model approach
- we will use expected model change maximization
    - for that, we need an approximate model, we will use the `SGDRegressor`
    
You can call `.fit()` on the `ASLearner`, which will by default fit the training data supplied. To select new units from the pool, you just need to call the `query()` method, which will return the selected `X` and the `query_ix` of these units. `ASLearner` expects the `n2` argument, which tells  how many units are queried at once. For sequential AL, we can set this to 1. Additionally, some query strategies can require different treatment effect estimates - EMCM needs uncertainty around the ITE. We can explicitly tell the the `ITEEstimator` to return all the predicted treatment effects. 
Then, we can teach the newly acquired units to the learner, by calling the `teach` function. The `score` function provides an evaluation of the given learner.

```python
from xbart import XBART
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
```

```python
X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(X, t, y, test_size=0.8,
                                                                    random_state=1005)
X_pool = np.copy(X_test)
asl = ASLearner(estimator = ITEEstimator(model = XBART(),
                                         two_model=False), 
         query_strategy=expected_model_change_maximization,
         X_training = X_train,
         t_training = t_train,
         y_training = y_train,
         X_pool     = X_pool,
         X_test     = X_test,
         approx_model=SGDRegressor())
asl.fit()
X_new, query_idx = asl.query(asl.X_pool, n2=10, return_mean = False)
asl.teach(X_new, t_test[query_idx], y_test[query_idx])
preds, *rest = asl.predict(asl.X_test)
asl.score(preds, y_test, t_test)
```




    0.190240308680396


