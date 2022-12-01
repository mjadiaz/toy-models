import numpy as np
from typing import Callable, Any, Dict

REWARD_FUNCTIONS: Dict[str, Callable[..., Any]] = dict()

def register(func: Callable[..., Any]) -> Callable[..., Any]:
    REWARD_FUNCTIONS[func.__name__] = func
    return func

@register
def exponential_density(lh, lh_factor, d, d_factor)->float:
    density = d*d_factor
    likelihood = lh*lh_factor 
    reward = likelihood*np.exp(-density)
    return reward

@register 
def density_difference(
        lh, lh_factor, d_tm1, d_t, d_factor 
        ) -> float:
    likelihood = lh*lh_factor 
    delta_density = d_tm1 - d_factor*d_t
    reward = likelihood*delta_density
    return reward

@register
def exponential_density_restricted(lh, lh_factor, d, d_factor)->float:
    density = d*d_factor
    likelihood = lh*lh_factor 
    r_exp_den = likelihood*np.exp(-density)
    reward = r_exp_den*(1+np.sign(lh))/2 \
                + lh*(1-np.sign(lh))/2
    return reward

