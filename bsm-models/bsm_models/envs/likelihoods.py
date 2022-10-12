import numpy as np
from typing import Callable, Any, Dict

LIKELIHOODS: Dict[str, Callable[..., Any]] = dict()

def register(func: Callable[..., Any]) -> Callable[..., Any]:
    LIKELIHOODS[func.__name__] = func
    return func

@register
def gaussian(variable, goal, sigma=10):
    lh = np.exp(- (variable-goal)**2/(2*sigma**2))
    return lh
@register
def heaviside(variable, goal):
    """Inverted heaviside to focus on points lower than goal"""
    lh = 1 - np.heaviside(variable-goal,0)
    return lh

