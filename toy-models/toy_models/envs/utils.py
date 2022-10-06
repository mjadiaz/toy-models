import io
from PIL import Image, ImageDraw
import imageio 
import numpy as np


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def save_mp4(images: list, name: str, fps=30):
    imageio.mimwrite(name, images, fps=fps)

def minmax(x, domain, codomain, reverse=False):
    '''
    Normalize an x value given a minimum and maximum mapping.

    Args:
    ----
    x = value to normalize
    domain = tuple for domain (from, to)
    codomain = tuple for codomain (from,to)
    reverse = inverse mapping, boolean.

    Returns:
    -------
    x_normalized (x_unormalized if reverse=True)
    '''
    min_dom, max_dom = domain
    min_codom, max_codom = codomain
    A,B = max_codom - min_codom, min_codom
    if reverse:
        return np.clip((x - B)*(max_dom-min_dom)/A + min_dom, min_dom, max_dom)
    else:
        return np.clip(A*(x - min_dom)/(max_dom-min_dom)+B, min_codom, max_codom)

