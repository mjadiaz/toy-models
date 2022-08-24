import gym
from gym.utils import seeding
from gym import spaces
from omegaconf import OmegaConf, DictConfig

import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.neighbors import KernelDensity
from PIL import Image, ImageDraw
import io
import imageio

jtplot.style()
plt.rcParams['axes.grid'] = False
plt.rc('axes', unicode_minus=False)


# Color maps
COLORS =['gist_earth', 'turbo' ]
# Utils

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

# Functions to use as toy simulators
def egg_box(x1,x2):
    '''
    Args: x1, x2
    Returns: observable
    '''
    model = lambda x1, x2: (2 + np.cos(x1/2)*np.cos(x2/2))**5
    obs = model(x1,x2)
    return obs

def gaussian_2d(x,y,mu1=13.5,mu2=13.5,height=100):
    '''
    Args: x1, x2
    Returns: observable
    '''
    A = height # Gaussian height
    x_0, y_0 = mu1, mu2 # Gaussian centers
    sigma_x, sigma_y = 0.5, 0.5

    gaussian = A * np.exp(-(((x - x_0)**2 / (2 * sigma_x**2)) + ((y - y_0)**2 / (2 * sigma_y**2))))
    return gaussian

def double_gaussian(x,y): 
    '''
    Args: x1, x2
    Returns: observable
    '''
    gaussian = gaussian_2d(x,y)\
            + gaussian_2d(x,y,mu1=11.5,mu2=11.5,height=100)
    return gaussian

FUNCTIONS = {
        'egg_box': egg_box,
        'double_gaussian': double_gaussian
        }

TF2D_DEFAULT_CONFIG = OmegaConf.create({
    'max_steps': 200,
    'd_factor': 5,
    'lh_factor': 5,
    'ps_bottom': 10,
    'ps_top': 15,
    'parameters_dimension': 2,
    'observables_dimension': 1,
    'function_name': 'egg_box',
    'goal': 100,
    'kernel_bandwidth': 0.4,
    'density_limit': 1,
    'kernel':  'epanechnikov',
    'norm_min': -1,
    'norm_max': 1,
    'parameter_shift_mode': False,
    'density_limit': 1.,
    'density_state': True,
    # Implement this below
    'observables_state': True,
    'parameters_state': True,
    'lh_function': 'distance',
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
        self.lh_function = self.config.lh_function
        
    
    def distance_likelihood(self,variable, maximum=1,acceptance=20):
        stability = acceptance**2*np.exp(-maximum)
        dlh = -np.log(((variable-self.goal)**2+stability)*(acceptance)**(-2))
        return dlh

    def gaussian_likelihood(self, variable, sigma=10):
        lh = np.exp(- (variable-self.goal)**2/(2*sigma**2))
        return 

    def likelihood(self, variable):
        if self.lh_function == 'gaussian':
            return self.gaussian_likelihood(variable)
        if self.lh_function == 'distance':
            return self.distance_likelihood(variable)
    
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
        
        if self.parameters_state: 
            ## Observation space: Pure parameter state
            self.observation_space = spaces.Box(
                 np.array([self.norm_min, self.norm_min]).astype(np.float32),#,0]).astype(np.float32),
                 np.array([self.norm_max, self.norm_max]).astype(np.float32),#np.inf]).astype(np.float32)
                 )
        if self.density_state and self.parameters_state:
            # Observation space: State including the density
            self.observation_space = spaces.Box(
                    np.array([self.norm_min, self.norm_min,0]).astype(np.float32),
                    np.array([self.norm_max, self.norm_max,np.inf]).astype(np.float32)
                    )

        if self.observables_state:
            self.observation_space = spaces.Box(
                    np.array([-np.inf]).astype(np.float32),
                    np.array([np.inf]).astype(np.float32)
                    )

        if self.observables_state and self.density_state:
            self.observation_space = spaces.Box(
                    np.array([0, -np.inf]).astype(np.float32),
                    np.array([np.inf, np.inf]).astype(np.float32)
                    )

        if self.observables_state and self.parameters_state:
            self.observation_space = spaces.Box(
                    np.array([self.norm_min, self.norm_min, -np.inf]).astype(np.float32),
                    np.array([self.norm_max, self.norm_max, np.inf]).astype(np.float32)
                    )

        if self.observables_state and self.parameters_state and\
                self.density_state:
            self.observation_space = spaces.Box(
                    np.array([self.norm_min, self.norm_min,0, -np.inf]).astype(np.float32),
                    np.array([self.norm_max, self.norm_max,np.inf, np.inf]).astype(np.float32)
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

        # Sample a random starting action 
        initial_params_real = self.parameter_space.sample()
        initial_params_norm = minmax(
                    initial_params_real, 
                    [self.ps_bottom, self.ps_top],
                    [self.norm_min, self.norm_max]
                    )
        
        # Starting state: Pure parameter state
        #self.state_real = np.hstack(initial_params_real)
        #self.state = np.hstack(initial_params_norm)
        # Starting with zero density
        #if self.density_state:
        #    self.state_real = np.hstack((initial_params_real, 0.))
        #    self.state = np.hstack((initial_params_norm, 0.))
        #else:
        #    self.state_real = np.hstack((initial_params_real))
        #    self.state = np.hstack((initial_params_norm))
       
        self.state_real = np.array([])
        self.state = np.array([])
        if self.parameters_state:
            self.state_real = np.hstack((self.state_real, initial_params_real))
            self.state = np.hstack((self.state, initial_params_norm))
        if self.observables_state:
            lh = self.lh_factor*self.simulator.run(*initial_params_real)
            self.observables_current = np.array([lh]).flatten()
            self.state_real = np.hstack((self.state_real, self.observables_current))
            self.state = np.hstack((self.state, self.observables_current))
        if self.density_state:
            #density = self.d_factor*self.density
            density = 0
            self.state_real = np.hstack((self.state_real, density))
            self.state = np.hstack((self.state, density))
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

    def get_reward(self, parameters=None):
        '''
        If parameters is None returns a single reward for the current state. 
        If parameters is not None, it must be np.ndarray with shape (n_points,2), so 
        that it can be used externaly for plotting.

        Returns:
        -------
        if parameters is None: Scalar reward. Else, array of reward values.

        '''

        reward = 0

        if parameters is None:
            lh = self.lh_factor*self.simulator.run(*self.next_params_real)
            density = self.d_factor*self.density
            rho = np.exp(-density)
            reward = lh*(rho*(1+np.sign(lh))+1-np.sign(lh))/2\
                    - self.density*(1+np.sign(np.log(self.density+(1-self.density_limit))))/2
            reward = reward[0]
        else:
            lh = self.lh_factor*self.simulator.run(parameters[:,0], parameters[:,1])
            densities = self.d_factor*self.predict_density(parameters)
            rho = np.exp(-densities)
            reward = lh*(rho*(1+np.sign(lh))+1-np.sign(lh))/2\
                    - densities*(1+np.sign(np.log(densities+(1-self.density_limit))))/2

        return reward


            


    
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
        self.density = self.predict_density(self.next_params_real.reshape(1,2))

        ## Get next state: Pure parameter state
        #self.state_real = np.hstack(self.next_params_real)#, self.density])
        #self.state = np.hstack(self.next_params)#, self.density])
        # Get next state: Density state
        #if self.density_state:
        #    self.state_real = np.hstack((self.next_params_real, self.density))
        #    self.state = np.hstack((self.next_params, self.density))
        #else:
        #    self.state_real = np.hstack((self.next_params_real))
        #    self.state = np.hstack((self.next_params))
        
        self.state_real = np.array([])
        self.state = np.array([])
        if self.parameters_state:
            self.state_real = np.hstack((self.state_real, self.next_params_real))
            self.state = np.hstack((self.state, self.next_params_real))
        if self.observables_state:
            lh = self.lh_factor*self.simulator.run(*self.next_params_real)
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

        return [self.state, self.reward, self.done, self.info]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def close(self):
        pass

    def render(self, mode = 'human', video_name='test'):
        likelihood = self.simulator.run(self.x, self.y)
        density = self.predict_density(self.xy)
        reward = self.get_reward(self.xy)

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


# Visualization 

class DensityPlot:
    def __init__(self, 
            x,
            y,
            X,
            Y,
            likelihood,
            density,
            reward,
            mode,
            video_name
            ):
        '''
        Generate plots in each step for the likelihood, the estimated density
        and the reward evolution.
        '''

        self.mode = mode
        self.video_name = video_name
        self.img_list = []
        self.fig, self.ax = plt.subplots(1,3, figsize=(20,4))

        self.lh_ax = self.ax[0]
        self.density_ax = self.ax[1]
        self.reward_ax = self.ax[2]

        self.cb1 = None
        self.cb2 = None
        self.cb3 = None

        self.frame = 0
        self.first_state = False

        plt.tight_layout()
              
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, 
                            top=0.90, wspace=0.4, hspace=0)

        # Save generated points to use in each frame
        self.x = x
        self.y = y 
        self.X = X
        self.Y = Y

        self.render_lh(likelihood)
        self.render_density(density)
        self.render_reward(reward)
        
        if self.mode == 'human':
            plt.show(block=False)
        
    
    def render(self, likelihood, density, reward, history, done):
        self.render_lh(likelihood, history)
        self.render_density(density, history)
        self.render_reward(reward, history)
        self.frame += 1
        
        if self.mode == 'human': 
            plt.pause(0.00001)
        if self.mode == 'video':
            img = fig2img(self.fig)
            self.img_list.append(img)
            if done:
                save_mp4(self.img_list, self.video_name)
    

            

    def render_lh(self, likelihood, history=None):
        '''Plot fixed likelihood'''
        self.lh_ax.clear()
        if self.cb1 is not None:
            self.cb1.remove()
        self.lh_ax.title.set_text('Likelihood')
        cp1 = self.lh_ax.scatter(self.X, self.Y, c=likelihood, cmap=COLORS[0])
        divider = make_axes_locatable(self.lh_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb1 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')
 
        if history is not None:
            self.lh_ax.scatter(history[:,0], history[:,1], color='orangered', s=20)

    def render_density(self, density, history=None):
        '''Plot initial density'''
        self.density_ax.clear()
        if self.cb2 is not None:
            self.cb2.remove()
        self.density_ax.title.set_text('Density')
        cp1 = self.density_ax.scatter(self.X, self.Y, c=density, cmap=COLORS[0])
        divider = make_axes_locatable(self.density_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb2 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')

    def render_reward(self, reward, history=None):
        '''Plot initial reward'''
        self.reward_ax.clear()
        if self.cb3 is not None:
            self.cb3.remove()
        self.reward_ax.title.set_text('Reward')
        cp1 = self.reward_ax.scatter(self.X, self.Y, c=reward, cmap=COLORS[1])
        divider = make_axes_locatable(self.reward_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb3 = self.fig.colorbar(cp1, cax=cax, orientation='vertical')



    def add_colorbar(self, ax, cb, plot):
        if cb is not None:
            cb.remove()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = self.fig.colorbar(plot, cax=cax, orientation='vertical')




