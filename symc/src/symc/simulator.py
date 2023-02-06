from omegaconf import OmegaConf, DictConfig
import numpy as np
from symc.test_functions import FUNCTIONS
from symc.utils import norm_map, vector_minmax, norm_data
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass#(slots=True)
class SyncConfig:
    """Default configuration for SyncSymulation"""
    simulation: str = 'egg_box'
    input_dimension: int = 2
    output_dimension: int = 1
    bounds: tuple = ((10.,15.))
    normalised_input: bool =  False
    norm_limits: tuple = ((-1,1))
    sim_config: Optional[dict] = None

@dataclass
class SimConfig:
    pass
     


class SyncSimulation:
    def __init__(self, 
                 sync_config: type[SyncConfig] = None, 
                 sync_simulator: Callable[[np.ndarray], np.ndarray] = None,
                 simulation_config: type[SimConfig] = None
                 ):
        if sync_config is None: 
            self.sync_config = SyncConfig()   
        else:
            self.sync_config = sync_config
        self.sim_config = simulation_config 
        self.sim_name = self.sync_config.simulation
        self.input_dimension = self.sync_config.input_dimension
        self.output_dimension = self.sync_config.output_dimension

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
        self.bounds = np.array(
            self.sync_config.bounds).reshape(-1,2)
        
        # If bounds are the same in all the dimensions
        if (len(self.bounds) == 1) and (self.input_dimension > 1):
            self.bounds = np.repeat(
                    self.bounds,
                    self.input_dimension,
                    axis=0
                    )


        self.normalised_input = self.sync_config.normalised_input
        self.norm_limits = np.array(
            self.sync_config.norm_limits).reshape(-1,2)
        self.norm_limits = np.repeat(
                    self.norm_limits, 
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

