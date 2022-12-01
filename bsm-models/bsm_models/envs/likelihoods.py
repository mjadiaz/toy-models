import numpy as np
from typing import Callable, Any, Dict
from bsm_models.envs.utils import minmax

LIKELIHOODS: Dict[str, Callable[..., Any]] = dict()

def register(func: Callable[..., Any]) -> Callable[..., Any]:
    LIKELIHOODS[func.__name__] = func
    return func

@register
def gaussian(variable, goal, sigma=10, width=None):
    lh = np.exp(- (variable-goal)**2/(2*sigma**2))
    return lh
@register
def heaviside(variable, goal):
    """Inverted heaviside to focus on points lower than goal"""
    lh = 1 - np.heaviside(variable-goal,0)
    return lh

@register
def sigmoid(x, goal, width=1):
    """
    For HS tau,center = 14, 180
    For HB tau,center = 0.3, 3
    """
    sigmoid = 1/(1+np.exp((x-goal)/width))
    return minmax(sigmoid, [0,1], [0,1])

@register 
def log(p, goal=0, width=10):
    log_lh = 1/np.log((goal-p)**2/width+ np.exp(1))
    return minmax(log_lh, [0,1], [-1,1])
