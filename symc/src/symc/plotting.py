import pandas as pd
import seaborn as sns
import numpy as np
sns.set_theme(style='ticks')
sns.set_palette('Blues_r')
sns.set_style({'font.family': 'serif', 'serif':['Century']})

def create_dataframe(x,labels: list[str]):
    data = {}
    for i,name in enumerate(labels):
        data[name] = x[:,i]
    df = pd.DataFrame(data)
    return df

    
CORNER = dict(
            diag_kind="kde",
            corner=True, 
            plot_kws=dict(marker=".", s=5, edgecolor="none"), 
            diag_kws=dict(fill=False),
            height=2.5
            )

def corner_plot( x, labels, config=CORNER):
    if isinstance(x, pd.DataFrame):
        df = x
    else:
        df = create_dataframe(x, labels)
    g = sns.pairplot(df, **config)
    return g

def color_plot( x, y, labels, color_label=None, cmap='Blues_r',config=CORNER):
    config['plot_kws']['c'] = y
    config['plot_kws']['cmap'] = cmap
    g = corner_plot(x=x, labels=labels, config=config)
    return g
    

