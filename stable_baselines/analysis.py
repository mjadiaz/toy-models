from omegaconf import OmegaConf, DictConfig
import numpy as np
import os
import torch
import gym



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


def generate_action_space(action_dim, bottom, top, samples=50):
    action_space = np.random.uniform(-1, 1 ,size=(samples**action_dim, 2))
    return action_space

def generate_state_action_batch(state, action_dim, bottom=-1., top=1., samples=50):
    actions = generate_action_space(action_dim, bottom, top, samples=samples)
    state = state.reshape(1,-1)
    states = np.repeat(state, int(samples**action_dim),axis=0)
    return states, actions
    


class PolicyNetwork:
    def __init__(
            self,
            actor,
            env
            ):

        self.action_dimension = env.action_space.shape[0]
        self.state_dimension = env.observation_space.shape[0]
        self.actor = actor

    def get_action(self, state):
        print(state.shape)
        state = torch.tensor(state).float().to(self.actor.device)
        # Get predictions
        action = self.actor(state)
        action = action.detach().cpu().numpy()
        return action

    def get_action_space(self, samples=100,state_space=None):
        if state_space is None:
            state_space = np.random.uniform(-1, 1 ,size=(samples, self.action_dimension))
        else:
            state_space = state_space
        self.current_states = state_space
        states = torch.tensor(state_space).float().to(self.actor.device)
        # get predictions
        actions = self.actor(states)
        actions = actions.detach().cpu().numpy()
        self.current_actions = actions
        return state_space, actions

    def generate_trajectories(self, state_space=None, steps=10, samples=100):
        if state_space is None:
            state_space = np.random.uniform(-1, 1 ,size=(self.samples, 2))
        states = torch.tensor(state_space).float().to(self.actor.device)
        # get predictions
        actions_trajectory = [state_space]
        for n in range(steps):
            actions = self.actor(states)
            actions = actions.detach().cpu().numpy()
            actions_trajectory.append(actions)
        return actions_trajectory

class QNetwork:
    def __init__(
            self,
            critic,
            env,
            ):

        self.critic = critic
        self.action_dimension = env.action_space.shape[0]
        self.state_dimension = env.observation_space.shape[0]
        self.samples = 100
        
    
    # Generate batch 
    def get_q_values(self, state):
        states, actions = generate_state_action_batch(
                state, 
                action_dim=self.action_dimension,
                samples=self.samples
                )
        self.current_states = states
        self.current_actions = actions
        states = torch.tensor(states).float().to(self.critic.device)
        actions = torch.tensor(actions).float().to(self.critic.device)
        
        # Get predictions
        q_values = self.critic(states, actions)[0]
        q_values = q_values.detach().cpu().numpy()
        self.current_q_values = q_values
        return q_values

    def random_state_q_values(self):
        state = np.random.uniform(-1,1,size=self.state_dimension)
        q_values = self.get_q_values(state)
        self.current_state = state
        return q_values

