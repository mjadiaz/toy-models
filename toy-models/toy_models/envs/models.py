import numpy as np
from typing import Callable, Any, Dict


FUNCTIONS: Dict[str, Callable[..., Any]] = dict()


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    FUNCTIONS[func.__name__] = func
    return func

@register
def egg_box(x1,x2):
    '''
    Args: x1, x2
    Returns: observable
    '''
    model = lambda x1, x2: (2 + np.cos(x1/2)*np.cos(x2/2))**5
    obs = model(x1,x2)
    return obs

@register
def gaussian_2d(x,y,mu1=13.5,mu2=13.5,height=100):
    '''
    Args: x1, x2
    Returns: observable
    '''
    A = height # Gaussian height
    x_0, y_0 = mu1, mu2 # Gaussian centers
    sigma_x, sigma_y = 0.5, 0.5

    gaussian = A * np.exp(-(((x - x_0)**2 / (2 * sigma_x**2)) + ((y - y_0)**2 / (2 * sigma_y**2))))
    return gaussian

@register 
def double_gaussian(x,y): 
    '''
    Args: x1, x2
    Returns: observable
    '''
    gaussian = gaussian_2d(x,y)\
            + gaussian_2d(x,y,mu1=11.5,mu2=11.5,height=100)
    return gaussian



