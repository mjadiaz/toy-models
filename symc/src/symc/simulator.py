from omegaconf import OmegaConf, DictConfig
import numpy as np
from symc.test_functions import FUNCTIONS
from symc.utils import norm_map, vector_minmax, norm_data


SYNC_CONFIG = OmegaConf.create({
    'simulation': 'egg_box',
    'input_dimension': 2,
    'output_dimension': 1,
    'bounds': [[10., 15.]],
    'normalised_input':  False,
    'norm_limits': [[-1,1]],
    'sim_config': None
    })

class Simulator:
    def __init__(self, sync_config=SYNC_CONFIG, sync_simulator= None,  internal_config=None):
        self.config = sync_config
        self.sim_config = internal_config
        self.sim_name = self.config.simulation
        self.input_dimension = self.config.input_dimension
        self.output_dimension = self.config.output_dimension

        if sync_simulator is None:
            self._function = FUNCTIONS[self.sim_name]
            self.function = lambda x: self._function(
                x, 
                self.input_dimension,
                config=self.sim_config
                )
        else: 
            self._function = lambda x: sim_model(
                    x,
                    self.input_dimension,
                    config=self.sim_config
                    )
        self.bounds = np.array(self.config.bounds)
        
        # If bounds are the same in all the dimensions
        if (len(self.bounds) == 1) and (self.input_dimension > 1):
            self.bounds = np.repeat(
                    self.bounds,
                    self.input_dimension,
                    axis=0
                    )


        self.normalised_input = self.config.normalised_input
        self.norm_limits = np.repeat(
                    np.array(self.config.norm_limits), 
                    self.input_dimension, 
                    axis=0)
    
    def __repr__(self):
        return f'{self.input_dimension} dimensional {self.sim_name} simulator.'

    def __call__(self, x):
        if self.normalised_input:
            x = self.normalise(x, reverse=True)
        return self.function(x)
    
    def normalise(self, x, reverse=False):
        '''
        Normalise the input vector.
        '''

        normalised_x = norm_data(
            x,
            self.bounds,
            self.norm_limits,
            reverse=reverse
            )
        return normalised_x
    
    def sample(self, size):
        samples = np.random.uniform(
                self.bounds[:,0], 
                self.bounds[:,1], 
                size=(size, self.input_dimension)
                )
        return samples


