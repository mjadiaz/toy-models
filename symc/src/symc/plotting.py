import pandas as pd
import seaborn as sns
import numpy as np
from copy import deepcopy

sns.set_theme(style='ticks')
sns.set_palette('Blues_r')
sns.set_style({'font.family': 'serif', 'serif':['Century']})

def create_dataframe(x,labels: list[str]):
    data = {}
    for i,name in enumerate(labels):
        data[name] = x[:,i]
    df = pd.DataFrame(data)
    return df

    
_CORNER = dict(
            diag_kind="kde",
            corner=True, 
            plot_kws=dict(marker=".", s=5, edgecolor="none"), 
            diag_kws=dict(fill=False),
            height=2.5
            )
CORNER_CONFIG = deepcopy(_CORNER)

def mono_corner_plot( x, labels=None, config=None):
    config = deepcopy(_CORNER) if config is None else config
    if isinstance(x, pd.DataFrame):
        df = x
    else:
        if labels is None:
            labels = [r'$\theta_{}$'.format(i) for i in range(x.shape[-1])]
        df = create_dataframe(x, labels)

    g = sns.pairplot(df, **config)
    return g

def color_corner_plot( x, y, labels=None, color_label=None, cmap='Blues_r',config=None):
    config = deepcopy(_CORNER) if config is None else config
    config['plot_kws']['c'] = y
    config['plot_kws']['cmap'] = cmap
    g = corner_plot(x=x, labels=labels, config=config)
    return g

def corner_plot( x, y=None, labels=None, color_label=None, cmap='Blues_r',config=None):
    if y is None:
        mono_corner_plot(x=x, labels=labels, config=config)
    else:
        color_corner_plot(
                x=x, 
                y=y, 
                labels=labels, 
                color_label=color_label, 
                cmap=cmap, 
                config=config
                )
    

