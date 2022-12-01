import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import softmax
import time
import imageio

import torch
from PIL import Image, ImageDraw

from stable_baselines3 import DDPG, TD3
#import pathlib
from toy_models.envs.utils import minmax

from scripts.q_integration import mc_partial_integration
from scripts.q_integration import mc_total_integration
from scripts.q_integration import q_values_given

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


#d = pathlib.Path("logs/")
## iterate directory
#res = []
#for entry in d.iterdir():
#    # check if it a file
#    if entry.is_file():
#        res.append(entry)
        
#selection = st.sidebar.selectbox("Select", res)
#st.sidebar.write('Displaying ', selection)

selection = st.sidebar.text_input(
        'Path to saved TD3 model', 
        value="./logs/test/final.zip",
        )
model = TD3.load(selection)



critic = model.critic
actor = model.policy


tab1, tab2, tab3, tab4 = st.tabs([
    "Q-Network", 
    "Policy", 
    'Total Monte Carlo Integration', 
    'Partial Monte Carlo Integration'])


with tab3:
    st.write('# Q-Value Total Integration')
    integration = []
    x = np.logspace(2, 5, num=100)
    x = np.array([int(i) for i in x])
    for i in x:
        integ = mc_total_integration(critic, n_points=i)
        integration.append(integ)
    fig, ax = plt.subplots()
    ax.plot(x,integration)
    st.pyplot(fig)

with tab4:
    n_samples = st.slider(
            'Number of samples', 
            min_value=10, 
            max_value=1000, 
            value=10, 
            step=int((1000-10)/10), 
            )
    #n_samples =100
    states = np.zeros((n_samples,2))
    integrated_q = np.zeros((n_samples,1))
    
    my_bar = st.progress(0)

    for i in range(n_samples):
        state, integ = mc_partial_integration(critic, n_points=int(1e4))
        states[i] = state
        integrated_q[i] = integ
        my_bar.progress((i+1)/n_samples)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.scatter(states[:,0], states[:,1], c=integrated_q, s=5)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([-1.05, 1.05])

    st.pyplot(fig)

with tab1:

    samples = 100000
    
    state, next_states, q_values = q_values_given(critic, samples)
    fig, ax = plt.subplots(figsize = (4,4))
    
    ax.title.set_text('Critic (Q network)')
    cp1 = ax.scatter(next_states[:,0], next_states[:,1], c=q_values, s=4, cmap='Blues', label='space of posible actions')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')
    ax.scatter(state[0], state[1], c='Orange',s=20, label='Initial Random state')
    #ax.scatter(actions[max_q_index,0], actions[max_q_index,1], c='Orange',s=20, label='State with Max Q')
    #ax.scatter(policy_next_state[0], policy_next_state[1],c='Cyan',s=20, label='Next State from Policy')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    
    
    text = '''
    # Q-network analysis
    For a random state in orange, the plot shows all the possible actions with their corresponding Q-value.
    
    Reload the page to check a new random state.
    '''
    
    st.markdown( text)
    st.pyplot(fig)
    

     
     
with tab2:
    #text = '''
    ## Q-network analysis
    #Loads the saved Q-network (critic) model from the checkpoint specified in `train.py` script.
    #For a random state in white, the plot shows all the possible actions with their corresponding Q-value.
    #
    #Reload the page to check a new random state.
    #'''
    #
    #st.markdown( text)
    samples = 100000
    states = np.random.uniform(-1,1,size=(samples, 2))
    state_space =  torch.tensor(states).float()
    images = []
    fig, ax = plt.subplots(figsize = (4,4))
    #
    ax.title.set_text(f'Actor - Policy at step 0')
    ax.scatter(state_space[:,0], state_space[:,1], c='Orange',s=1, label='States')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    img = fig2img(fig)
    images.append(img)
    for t in range(1,30):
        state_space =  torch.tensor(state_space).float()
        state_tp1 = actor(state_space)
        state_tp1 = state_tp1.detach().numpy()

        fig, ax = plt.subplots(figsize = (4,4))
        #
        ax.title.set_text(f'Actor - Policy at step {t}')
        ax.scatter(state_tp1[:,0], state_tp1[:,1], c='Orange',s=1, label='States')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        img = fig2img(fig)
        images.append(img)
        state_space = state_tp1

    

    imageio.mimwrite('random_n_policy.mp4', images, fps=5, quality=10)
    st.video('random_n_policy.mp4', format="video/mp4", start_time=0)



