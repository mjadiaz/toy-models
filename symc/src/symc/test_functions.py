import numpy as np
from typing import Callable, Any, Dict


FUNCTIONS: Dict[str, Callable[..., Any]] = dict()


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    FUNCTIONS[func.__name__] = func
    return func

@register
def egg_box(x, dimension, config=None):
    x = x.reshape(-1,dimension)  
    observation =  np.power(2 + np.prod(np.cos(x/2), axis=1),5)
    return observation.reshape(len(x),-1)

@register
def rosenbrock2d(x, dimension, config=None):
    x = x.reshape(-1, dimension)
    f = np.zeros(x.shape[0])
    for i in range(dimension-1):
        f += (1-x[:,i])**2 + 100*(x[:,i+1] - x[:,i]**2)**2
    return f.reshape(len(x),1)

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



