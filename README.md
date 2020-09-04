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
X = np.random.normal(size = 1000).reshape((500,2))
t = np.random.binomial(n = 1, p = 0.5, size = 500)
y = np.random.binomial(n = 1, p = 1/(1+np.exp(X[:, 1]*2 + t*3)))
a = ITEEstimator(LogisticRegression(), X, t, y)
```
