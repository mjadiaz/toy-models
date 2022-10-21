import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import softmax
import time
import imageio

import torch
from PIL import Image, ImageDraw

from train_sb3 import alg_name, env_config, env
from stable_baselines3 import DDPG

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

model_name = st.sidebar.text_input('Trained model name: ')
st.sidebar.write('Displaying ', model_name)
model = DDPG.load("test/"+model_name)

critic = model.critic
actor = model.policy


tab1, tab2, tab3, tab4 = st.tabs(["Q-Network", "Policy", 'Combined sampling', 'QA Sampling'])


with tab1:
    samples = 100000
    action_space = np.random.uniform(-1,1,size=(samples, 2))
    state_space = np.repeat(np.random.uniform(-1,1,size=(1,2)), samples, axis=0)
    state = state_space[0]
    next_states =  action_space
    action_space = torch.tensor(action_space).float()
    state_space =  torch.tensor(state_space).float()

    q_values = critic(state_space, action_space)
    q_values = q_values[0].flatten().detach().numpy()

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



