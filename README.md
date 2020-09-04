# Automatic Stopping for Batch-mode Experimentation
> Code for using active learning and automatic stopping for designing experiments


Created with nbdev by ZP

## Install

`python -m pip install  git+https://github.com/puhazoli/asbe`

## How to use

```python
from asbe.core import *
# say_hello("Zoltan")
```

```python
ASLearner()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-2-09d18ea934eb> in <module>
    ----> 1 ASLearner()
    

    TypeError: __init__() missing 3 required positional arguments: 'estimator', 'query_strategy', and 'assignment_fc'


```python
import numpy as np
```

```python
def daily_hun_count():
    return(np.random.poisson(100, 1))
```

```python
daily_hun_count()
```




    array([94])


