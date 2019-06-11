from __future__ import print_function
from __future__ import division

from sklearn.metrics import fbeta_score as sk_fbeta
from functools import wraps


__all__ = ['f05_score',
           'f04_score']

def fbeta_create(beta=1.0):
    def fbeta_wrapper(func):
        @wraps(func)
        def fbeta_score(*args, **kwargs):
            kwargs.update({"beta": beta})
            return sk_fbeta(*args, **kwargs)

        return fbeta_score
    return fbeta_wrapper


@fbeta_create(beta=0.5)
def f05_score(*args, **kwargs):
    pass

@fbeta_create(beta=0.4)
def f04_score(*args, **kwargs):
    pass
