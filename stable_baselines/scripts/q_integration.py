import numpy as np 
import torch
from scipy.special import softmax
from toy_models.envs.utils import minmax

def q_values_given(critic, n_points, state=None, given='state'):
    samples = n_points 
    action_space = np.random.uniform(-1,1,size=(samples, 2))
    if state is None: 
        state_space = np.repeat(np.random.uniform(-1,1,size=(1,2)), samples, axis=0)
    else:
        state_space = np.repeat(state.reshape((1,2)), samples, axis=0)
    state = state_space[0]
    next_states =  action_space
    action_space = torch.tensor(action_space).float()
    state_space =  torch.tensor(state_space).float()
    
    # It works because states and actions has the same dimentions
    # Be careful
    if given == 'state':
        q_values = critic(state_space, action_space)
        q_values = q_values[0].flatten().detach().numpy()
        actions = action_space.detach().numpy()
        return state, actions, q_values.reshape((n_points, 1))
    else:
        q_values = critic(action_space, state_space)
        q_values = q_values[0].flatten().detach().numpy()
        states = action_space.detach().numpy()
        action = state
        return action, states, q_values.reshape((n_points, 1))

def q_values_n_best(state, action_space, q_values, n_best=5):
    state_space = np.repeat(state.reshape((1,2)), len(q_values), axis=0)
    # Merge into one matrix
    actions_q_values = np.hstack([action_space, q_values])
    # Sort by q_values
    actions_q_values = actions_q_values[actions_q_values[:,-1].argsort()] 
    # Desending order and select first n_best
    actions_q_values = np.flip(actions_q_values, axis=0)[:n_best]
    # Define index
    actions_index = np.arange(len(actions_q_values)).reshape(-1,1)
    # Get Soft Max for q values
    soft_max_q = softmax(actions_q_values[:,-1])
    # Separate in individuals arrays
    actions = actions_q_values[:, :2]
    q_values = actions_q_values[:, -1]
    return  state, actions, q_values, soft_max_q, actions_index

def mc_total_integration( critic, n_points=100000, dim=4):
    a,b = -1,1
    action_space = np.random.uniform(a,b,size=(n_points,2)) 
    state_space = np.random.uniform(a,b,size=(n_points,2))

    q_values = critic(
            torch.tensor(state_space).float(),
            torch.tensor(action_space).float()
            )

    q_values = q_values[0].flatten().detach().numpy()

    soft_max_q = softmax(q_values)
    norm_data = minmax(
            q_values, 
            [q_values.min(), q_values.max()],
            [0,1])
    
    y_mean = np.sum(norm_data)/n_points
    integ = ((b-a)**dim)*y_mean

    return integ

def mc_partial_integration(critic, state = None, n_points=100000, dim=2):
    a,b = -1,1
    action, states, q_values = q_values_given(
            critic, n_points=n_points, state=state, given='action'
            ) 
    soft_max_q = softmax(q_values)
    norm_data = minmax(
            q_values, 
            [q_values.min(), q_values.max()],
            [0,1])
    
    y_mean = np.sum(q_values)/n_points
    integ = ((b-a)**dim)*y_mean
    
    return action, integ


def mc_partial_integration_best(critic, state = None, n_points=1000, dim=2, n_best=500):
    a,b = -1,1

    action, states, q_values, soft_max_q, actions_index = q_values_n_best(
        *q_values_given(critic, n_points=n_points, state=state, given='action'),
        n_best=n_best)
    norm_data = minmax(
            q_values, 
            [q_values.min(), q_values.max()],
            [0,1])
    y_mean = np.sum(q_values)/n_points
    integ = ((b-a)**dim)*y_mean
    
    return action, integ
