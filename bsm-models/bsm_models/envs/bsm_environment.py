import gym
from gym.utils import seeding
from gym import spaces
from omegaconf import OmegaConf, DictConfig

import numpy as np
import random
import string

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from bsm_models.envs.models import MODELS 
from bsm_models.envs.likelihoods import LIKELIHOODS 
from bsm_models.envs.rendering import DensityPlot 
from bsm_models.envs.utils import minmax 
from bsm_models.envs.reward_functions import REWARD_FUNCTIONS



PHENOENV_DEFAULT_CONFIG = OmegaConf.create({
    'max_steps': 200,
    'd_factor': 5,
    'lh_factor': 3,
    'ps_bottom': 10,
    'ps_top': 15,
    'goal': 100,
    'kernel_bandwidth': 0.3,
    'density_limit': 0.3,
    'kernel':  'gaussian',
    'norm_min': -0.5,
    'norm_max': 0.5,
    'parameter_shift_mode': True,
    'density_limit': 1.,
    'density_state': False,
    'observables_state': False,
    'parameters_state': True,
    'reward_function': 'exponential_density',
    'simulator_name': 'SPhenoHbHs'
        })

HEP_DEFAULT_CONFIG = OmegaConf.create(
    """
    model:
        name: 'BLSSM'
        neutral_higgs: 6
        charged_higgs: 1
        parameters:
            name:      ['m0', 'm12',   'a0', 'tanbeta']
            low_lim:   [100.,  1000., 1000.,        1.]
            high_lim:  [1000., 4500., 4000.,       60.]
            lhs:
                index: [1,        2,      5,         3]
                block: ['MINPAR', 'MINPAR', 'MINPAR', 'MINPAR']
        observation:
            name:      ['Mh(1)', 'Mh(2)', 'obsratio', 'csq(tot)']
        goal:
            name:      ['Mh(1)', 'Mh2(2)', 'obsratio', 'csq(tot)']
            value:     [    93.,     125.,         0.,       0.]
            lh_type:   ['gaussian', 'gaussian', 'heaviside', heaviside]
    directories:
        scan_dir: '/mainfs/scratch/mjad1g20/test_env'
        reference_lhs: '/scratch/mjad1g20/rlhep/runs/ddpg_tests/SPhenoBLSSM_input/LesHouches.in.Step'
        spheno: '/scratch/mjad1g20/HEP/SPHENO/SPheno-3.3.8'
        higgsbounds: '/scratch/mjad1g20/HEP/higgsbounds-5.10.1/build'
        madgraph: '/scratch/mjad1g20/HEP/MG5_aMC_v3_1_1'
        higgssignals: '/scratch/mjad1g20/HEP/higgssignals-2.6.2/build'
    """)


def id_generator(size=7, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class Simulator:
    def __init__(self, config, hep_config):
        self.name = self.config.simulator_name

        self.goal_value = self.hep_config.model.goal.value
        self.lh_type = self.hep_config.model.goal.lh_type
        

        self.random_id = id_generator()
        self.model = MODELS[self.name](self.random_id, hep_config)
        self.lh_functions = LIKELIHOODS

    def higgs_masses_likelihood(self, observables_array: np.ndarray) -> float:
        likelihood_total = 1
        goal_iter = zip(observables_array, self.goal_value, self.lh_type)
        for obs_value, goal_value, lh_type in goal_iter:
            likelihood_total *= self.lh_functions[lh_type](obs_value, goal_value)
        return likelihood_total

    def likelihood(self, observables_array: np.ndarray):
        return self.higgs_masses_likelihood(observables_array)
    
    def run(self, observables_array: np.ndarray):
        obs = self.model(observables_array)
        lh = self.likelihood(obs)
        return lh
    

    
class PhenoEnv_v3(gym.Env):
    def __init__(self, 
            env_config: DictConfig = PHENOENV_DEFAULT_CONFIG,
            hep_config: DictConfig = HEP_DEFAULT_CONFIG
            ):
        self.config = env_config
        self.hep_config = hep_config

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
        self.parameters_state = self.config.parameters_state

        self.observables_dimension = len(
                self.hep_config.model.observation.name
                )

        # Physical parameter space info
        self.ps_bottom = self.hep_config.model.parameters.low_lim
        self.ps_top = self.hep_config.model.parameters.high_lim
        self.parameters_dimension = len(
                self.hep_config.model.parameters.name
                )
        # Initialize observation dimension 
        self.observation_dimension = 0

        # Hep Dimensions

        # Observation dimension += n_observables
        if self.parameters_state:
            self.observation_dimension += self.parameters_dimension

        # Observation dimension += n_observables
        if self.observables_state:
            self.observation_dimension += self.observables_dimension

        # Observation dimension: n_parameters + density
        if self.density_state:
            self.observation_dimension += 1
        
        # Action dimension: n_parameters
        self.action_dimension = self.parameters_dimension
        self.n_parameters = self.parameters_dimension

        # Visualization
        self.visualization = None

        # Initialize simulator
        self.simulator = Simulator(self.config, self.hep_config)

        # Initialize kernel for density estimation
        self.kernel = KernelDensity(
                bandwidth=self.config.kernel_bandwidth,
                kernel=self.config.kernel
                )

        ## Non normalized action  
        self.parameter_space = spaces.Box(
                np.array(self.ps_bottom).astype(np.float32),
                np.array(self.ps_top).astype(np.float32),
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
        self.reward = 0
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
            lh = self.lh_factor*self.simulator.run(*initial_params_real)
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
        self.reward = self.get_reward()
         
        # Fit kernel
        self.fit_kernel()
        
        self.counter += 1

        if self.counter == self.max_steps:
            self.done = True
        if self.terminal:
            self.done = True

        if self.done:
            self.model.close()

        return [self.state, self.reward, self.done, self.info]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def close(self):
        self.model.close()

    def render(self, mode = 'rgb'):
        pass

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

        lh = self.simulator.run(*self.next_params_real)
        reward = self.reward_function(
                lh, self.lh_factor, self.density, self.d_factor
                )
        return reward
    
    def get_reward_plot(self, parameters: np.ndarray) -> np.ndarray:
        '''
        Reward function definition for plotting. Using the parameters array 
        to calculate the reward.
        '''
        lh = self.simulator.run(parameters[:,0], parameters[:,1])
        #logprob_tm1 = self.kernel_tm1.score_samples(parameters)
        #densities_tm1 = np.exp(logprob_tm1)
        densities = self.predict_density(parameters)
        reward_array = self.reward_function(
                lh, self.lh_factor, densities, self.d_factor
                )
        return reward_array
