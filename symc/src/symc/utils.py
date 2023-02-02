import io
#from PIL import Image, ImageDraw
#import imageio 
import numpy as np


#def fig2img(fig):
#    """Convert a Matplotlib figure to a PIL Image and return it"""
#    buf = io.BytesIO()
#    fig.savefig(buf)
#    buf.seek(0)
#    img = Image.open(buf)
#    return img
#def save_mp4(images: list, name: str, fps=30):
#    imageio.mimwrite(name, images, fps=fps)

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

def vector_minmax(x, domains, codomains, reverse=False):
    '''
    Normalise mapping from domains to codomains where each element in
    x has different ranges.
    '''
    norm = np.zeros_like(x)
    # Generate iterable
    var_dom_cod = zip(x, domains, codomains)
    for i, vdc in enumerate(var_dom_cod):
        var, dom, cod = vdc
        norm[i] = minmax(var, dom, cod, reverse)
    return norm

def norm_data(x, domains, codomains, reverse=False):
    norm = np.zeros_like(x)
    for i in range(len(x)):
        norm[i] = vector_minmax(x[i], domains, codomains, reverse)
    return norm
def norm_vmap(x, domains, codomains, reverse=False):
    '''Wrapping vector_minmax for codomain hyper square'''
    if len(codomains.shape) == 1:
        codomains = np.repeat(codomains.reshape(-1,2),len(x), axis=0)
    norm = vector_minmax(x, domains, codomains, reverse)
    return norm

def norm_map(x, domains, codomains, reverse=False):
    if len(domains.shape) == 1:
        return minmax(x,domains, codomains, reverse)
    else:
        return norm_vmap(x, domains, codomains, reverse)



