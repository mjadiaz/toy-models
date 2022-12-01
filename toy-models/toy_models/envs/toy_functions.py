import gym
from gym.utils import seeding
from gym import spaces
from omegaconf import OmegaConf, DictConfig

import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from toy_models.envs.models import FUNCTIONS
from toy_models.envs.rendering import DensityPlot 
from toy_models.envs.utils import minmax 
from toy_models.envs.reward_functions import REWARD_FUNCTIONS
from toy_models.envs.likelihoods import LIKELIHOODS

TF2D_DEFAULT_CONFIG = OmegaConf.create({
    'max_steps': 200,
    'd_factor': 1,
    'lh_factor': 1,
    'ps_bottom': 10,
    'ps_top': 15,
    'parameters_dimension': 2,
    'observables_dimension': 1,
    'function_name': 'egg_box',
    'goal': 100,
    'kernel_bandwidth': 0.2,
    'density_limit': 0.8,
    'kernel':  'gaussian',
    'norm_min': -1.,
    'norm_max': 1.,
    'parameter_shift_mode': False,
    'density_state': False,
    'observables_state': False,
    'parameters_state': True,
    'lh_function': 'gaussian',
    'reward_function': 'exponential_density'
        })


class Simulator:
    def __init__(self, config):
        self.config = config
        self.name = self.config.function_name
        self.function = FUNCTIONS[self.name]
        self.goal = self.config.goal
        self.bottom = self.config.ps_bottom
        self.top = self.config.ps_top
        self.norm_min = self.config.norm_min
        self.norm_max = self.config.norm_max
        self.lh_functions = LIKELIHOODS
        self.lh_name = self.config.lh_function
        
    
    def distance_likelihood(self,variable, maximum=1,acceptance=20):
        stability = acceptance**2*np.exp(-maximum)
        dlh = -np.log(((variable-self.goal)**2+stability)*(acceptance)**(-2))
        return dlh

    def gaussian_likelihood(self, variable, sigma=10):
        lh = np.exp(- (variable-self.goal)**2/(2*sigma**2))
        return lh

    def likelihood(self, variable):
        #if self.lh_function == 'gaussian':
        #    return self.gaussian_likelihood(variable)
        #if self.lh_function == 'distance':
        #    return self.distance_likelihood(variable)
        return  self.lh_functions[self.lh_name](variable, self.goal)
    
    def run(self, x1, x2):
        obs = self.function(x1,x2)
        lh = self.likelihood(obs)
        return lh
    
    def generate_space(self, samples=100):
        x = np.linspace(self.bottom, self.top , samples)
        y = np.linspace(self.bottom, self.top , samples)
        X,Y = np.meshgrid(x,y)
    
        X_norm = minmax(X, [self.bottom, self.top ], [self.norm_min,self.norm_max])
        Y_norm = minmax(Y, [self.bottom, self.top ], [self.norm_min,self.norm_max])
        return X.flatten(), Y.flatten(), X_norm.flatten(), Y_norm.flatten()
    
    def generate_random_space(self, samples=100):
        x = np.random.uniform(self.bottom, self.top , samples)
        y = np.random.uniform(self.bottom, self.top , samples)
    
        X_norm = minmax(x, [self.bottom, self.top ], [self.norm_min,self.norm_max])
        Y_norm = minmax(y, [self.bottom, self.top ], [self.norm_min,self.norm_max])
        return x.flatten(), y.flatten(), X_norm.flatten(), Y_norm.flatten()

    
       

    

class ToyFunction2d_v1(gym.Env):
    def __init__(self, env_config: DictConfig = TF2D_DEFAULT_CONFIG):
        self.config = env_config

        # Env information
        self.max_steps = self.config.max_steps
        self.d_factor = self.config.d_factor
        self.lh_factor = self.config.lh_factor

        self.norm_min = self.config.norm_min
        self.norm_max = self.config.norm_max
        self.parameter_shift_mode = self.config.parameter_shift_mode
        self.density_limit = self.config.density_limit
        self.density_state = self.config.density_state
        self.reward_function = REWARD_FUNCTIONS[self.config.reward_function]

        # Include observables in the state
        self.observables_state = self.config.observables_state
        self.observables_dimension = self.config.observables_dimension
        self.parameters_state = self.config.parameters_state
        
        # Physical parameter space info
        self.ps_bottom = self.config.ps_bottom
        self.ps_top = self.config.ps_top
       
        # Initialize observation dimension 
        self.observation_dimension = 0

        # Observation dimension += n_observables
        if self.parameters_state:
            self.observation_dimension += self.config.parameters_dimension

        # Observation dimension += n_observables
        if self.observables_state:
            self.observation_dimension += self.observables_dimension

        # Observation dimension: n_parameters + density
        if self.density_state:
            self.observation_dimension += 1
        
        # Action dimension: n_parameters
        self.action_dimension = self.config.parameters_dimension
        self.n_parameters = self.config.parameters_dimension

        # Visualization
        self.visualization = None

        # Initialize simulator
        self.simulator = Simulator(self.config)
        ## Generate space for plotting
        self.x, self.y, self.X, self.Y = self.simulator.generate_space()
        self.XY = np.hstack([
            self.X.reshape(len(self.X),1), 
            self.Y.reshape(len(self.Y),1)])
        self.xy = np.hstack([
            self.x.reshape(len(self.x),1), 
            self.y.reshape(len(self.y),1)])
        # Initialize kernel for density estimation
        self.kernel = KernelDensity(
                bandwidth=self.config.kernel_bandwidth,
                kernel=self.config.kernel
                )

        ## Non normalized action  
        self.parameter_space = spaces.Box(
                self.ps_bottom*np.ones(self.action_dimension).astype(np.float32),
                self.ps_top*np.ones(self.action_dimension).astype(np.float32),
                )

        ## Normalized action space
        self.action_space = spaces.Box(
                -np.ones(self.action_dimension).astype(np.float32),
                 np.ones(self.action_dimension).astype(np.float32)
                )

        space_low = []
        space_high = []

        if self.parameters_state: 
            ## Observation space: Pure parameter state
            [space_low.append(self.norm_min) for _ in range(self.n_parameters)]
            [space_high.append(self.norm_max) for _ in range(self.n_parameters)]
        
        if self.observables_state:
            space_low.append(-np.inf)
            space_high.append(np.inf)

        if self.density_state:
            space_low.append(0)
            space_high.append(np.inf)

        self.observation_space = spaces.Box(
                np.array(space_low).astype(np.float32),
                np.array(space_high).astype(np.float32)
                )

    
        self.seed()

    def reset(self):
        self.params_history = np.zeros((self.max_steps, self.n_parameters))
        self.params_history_real = np.zeros((self.max_steps, self.n_parameters))
        self.reward = -self.lh_factor
        self.done = False
        self.terminal = False
        self.info = dict()
        self.counter = 0
        self.density = 0.

        # Sample a random starting action 
        initial_params_real = self.parameter_space.sample()
        initial_params_norm = minmax(
                    initial_params_real, 
                    [self.ps_bottom, self.ps_top],
                    [self.norm_min, self.norm_max]
                    )
        
       
        self.state_real = np.array([]).astype(np.float32)
        self.state = np.array([]).astype(np.float32)
        
        if self.parameters_state:
            self.state_real = np.hstack((self.state_real, initial_params_real))
            self.state = np.hstack((self.state, initial_params_norm))
        if self.observables_state:
            lh = self.simulator.run(*initial_params_real)
            self.observables_current = np.array([lh]).flatten()
            self.state_real = np.hstack((self.state_real, self.observables_current))
            self.state = np.hstack((self.state, self.observables_current))
        if self.density_state:
            self.state_real = np.hstack((self.state_real, self.density))
            self.state = np.hstack((self.state, self.density))
        # Calculate initial kernel
        self.kernel.fit(initial_params_norm.reshape(1,2))

        return self.state
    
    def parameter_shift(self, action):
        '''
        Takes the current state and perform a shift
        
        Args:
        ----
        action: delta_parameter in the normalized space

        Returns:
        -------
        next_params_normalized
        next_params_real
        '''
        
        next_params = self.state[:2] + action
        next_params = np.clip(next_params,self.norm_min,self.norm_max)
        next_params_real = minmax(
                next_params,
                [self.ps_bottom, self.ps_top],
                [self.norm_min,self.norm_max],
                reverse=True
                )
        return next_params, next_params_real

    def direct_action(self, action):
        next_params = action
        next_params_real = minmax(
                next_params,
                [self.ps_bottom, self.ps_top],
                [self.norm_min,self.norm_max],
                reverse=True
                )
        return next_params, next_params_real

    def fit_kernel(self):
        '''
        Fit the kernel to the collected states acording to the counter.
        '''
        data = self.params_history_real[:self.counter+1]
        self.kernel_tm1 = self.kernel
        self.kernel.fit(data)

    def predict_density(self, x: np.ndarray):
        '''
        Estimates the density for the data x.
        
        Args:
        ----
        x: np.ndarray

        Returns:
        -------
        density: np.ndarray
        '''
        logprob = self.kernel.score_samples(x)
        density = np.exp(logprob)
        return density



    
    def step(self, action):
        '''
        Takes the action, perform the parameter shift in the state, 
        estimates the density of visited points and uses this 
        information to generate the next state.
        '''
        
        # Take action
        if self.parameter_shift_mode:
            self.next_params, self.next_params_real = self.parameter_shift(action)
        else: 
            self.next_params, self.next_params_real = self.direct_action(action)

        # Add to history
        self.params_history[self.counter] = self.next_params
        self.params_history_real[self.counter] = self.next_params_real
        # Estimate density
        self.density_tm1 = self.density
        self.density = self.predict_density(self.next_params_real.reshape(1,2))
        self.density = float(self.density[0])

        self.state_real = np.array([]).astype(np.float32)
        self.state = np.array([]).astype(np.float32)
        if self.parameters_state:
            self.state_real = np.hstack((self.state_real, self.next_params_real))
            self.state = np.hstack((self.state, self.next_params))
        if self.observables_state:
            lh = self.simulator.run(*self.next_params_real)
            self.observables_current = np.array([lh]).flatten()
            self.state_real = np.hstack((self.state_real, self.observables_current))
            self.state = np.hstack((self.state, self.observables_current))
        if self.density_state:
            self.state_real = np.hstack((self.state_real, self.density))
            self.state = np.hstack((self.state, self.density))

        # Get Reward
        #self.reward_tm1 = self.reward
        #self.reward_t = self.get_reward()
        #self.reward = self.reward_t - self.reward_tm1

        self.reward = self.get_reward()

        # Fit kernel
        self.fit_kernel()
        
        self.counter += 1

        if self.counter == self.max_steps:
            self.done = True
        if self.terminal:
            self.done = True

        return [self.state, self.reward, self.done, self.info]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def close(self):
        pass

    def render(self, mode = 'human', video_name='test'):
        likelihood = self.simulator.run(self.x, self.y)
        
        density = self.predict_density(self.xy)
        reward = self.get_reward_plot(self.xy)

        if self.visualization is None:
            self.visualization = DensityPlot(
                        self.x,
                        self.y,
                        self.X,
                        self.Y,
                        likelihood,
                        density,
                        reward,
                        mode,
                        video_name)
        else:
            self.visualization.render(
                    likelihood,
                    density,
                    reward,
                    self.params_history[:self.counter],
                    self.done
                    )

    def get_reward(self) -> float:
        '''
        If parameters is None returns a single reward for the current state. 
        If parameters is not None, it must be np.ndarray with shape (n_points,2), so 
        that it can be used externaly for plotting.

        Returns:
        -------
        reward: float
        '''

        reward = 0
        
        self.lh = self.simulator.run(*self.next_params_real)
        #reward = self.reward_function(
        #        lh, self.lh_factor, self.density_tm1, self.density, self.d_factor
        #        )
        reward = self.reward_function(
                self.lh, self.lh_factor, self.density, self.d_factor
                )
        #if self.density >= self.density_limit:
        #    reward += -self.lh_factor
        #    self.terminal = True

        return reward
    
    def get_reward_plot(self, parameters: np.ndarray) -> np.ndarray:
        '''
        Reward function definition for plotting. Using the parameters array 
        to calculate the reward.
        '''
        lh = self.simulator.run(parameters[:,0], parameters[:,1])
        logprob_tm1 = self.kernel_tm1.score_samples(parameters)
        densities_tm1 = np.exp(logprob_tm1)
        densities = self.predict_density(parameters)
        reward_array = self.reward_function(
                lh, self.lh_factor, densities, self.d_factor
                )
        return reward_array
