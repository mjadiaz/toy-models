from typing import Callable, Any, Dict

from hepaid.hepread import SLHA, LesHouches
from hepaid.hepread import HiggsBoundsResults, HiggsSignalsResults
from hepaid.heptools import Spheno, HiggsBounds, HiggsSignals
from omegaconf import OmegaConf, DictConfig

import numpy as np
import os
import ray
import shutil
import pandas as pd

MODELS: Dict[str, Callable[..., Any]] = dict()


def register(hep_stack: Callable[..., Any]) -> Callable[..., Any]:
    MODELS[hep_stack.__name__] = func
    return func

class Space:
    """
    Class function to imitate gym.Space (From HEP Scanner)
    """
    def __init__(self, config: DictConfig):
		self.block = config.model.parameters.lhs.block
		self.index = config.model.parameters.lhs.index
		self.low_lim = config.model.parameters.low_lim
		self.high_lim = config.model.parameters.high_lim

		self.sample_dim = len(self.index)

	def sample(self):
		sample_point = np.zeros(self.sample_dim)
		for i, limits in enumerate(zip(self.low_lim, self.high_lim)):
			low, high = limits
			sample_point[i] = np.random.uniform(low,high)
		return sample_point

@register
class SPhenoHbHs:
	def __init__(self,
			sampler_id: int,
			config: DictConfig,
			lhs: LesHouches = None
			):
		'''Initialize the HEP tools'''
		self.hp = config
		self.scan_dir = self.hp.directories.scan_dir
		self.sampler_id_dir = os.path.join(
			self.scan_dir, str(sampler_id)
			)

		# Parameter information 
		self.block = self.hp.model.parameters.lhs.block
		self.index = self.hp.model.parameters.lhs.index

		self.current_lhs = 'LesHouches.in.Step'
		self.current_spc = 'Spheno.spc.Step'

		self.space = Space(config)


		self.spheno = Spheno(
			spheno_dir = self.hp.directories.spheno,
			work_dir = self.sampler_id_dir,
			model_name = self.hp.model.name,
			)
		self.higgs_bounds = HiggsBounds(
			higgs_bounds_dir = self.hp.directories.higgsbounds,
			work_dir = self.sampler_id_dir,
			neutral_higgs = self.hp.model.neutral_higgs,
			charged_higgs = self.hp.model.charged_higgs,
			)
		self.higgs_signals = HiggsSignals(
			higgs_signals_dir = self.hp.directories.higgssignals,
			work_dir = self.sampler_id_dir,
			neutral_higgs = self.hp.model.neutral_higgs,
			charged_higgs = self.hp.model.charged_higgs,
			)
		if lhs is None:
			self.lhs = LesHouches(
				file_dir = self.hp.directories.reference_lhs,
				work_dir = self.sampler_id_dir,
				model =  self.hp.model.name,
				)
		else:
			self.lhs = lhs

	def create_dir(self, run_name: str):
		if not(os.path.exists(self.sampler_id_dir)):
			os.makedirs(self.sampler_id_dir)

	def get_lhs(self):
		return self.lhs

    def sample(self, parameter_point: np.ndarray) -> Dict[str, float]:

		param_card = None

		sample_point = parameter_point
		# To iterate on each block name, parameter index and
		# the sampled value respectively
		params_iterate = zip(
			self.block,
			self.index,
			self.hp.model.parameters.name,
			sample_point
			)
		parameters = {}
		for params in params_iterate:
			block, index, name, value = params
			self.lhs.block(block).set(index, value)
			parameters[name] = value

		# Create new lhs file with the parameter values
		self.lhs.new_file(self.current_lhs)
		# Run spheno with the new lhs file
		param_card, spheno_stdout = self.spheno.run(
			self.current_lhs,
			self.current_spc
			)
		if param_card is not None:
			self.higgs_bounds.run()
			self.higgs_signals.run()
			higgs_signals_results = HiggsSignalsResults(
				self.sampler_id_dir,
				model = self.hp.model.name
				).read()
			higgs_bounds_results = HiggsBoundsResults(
				self.sampler_id_dir,
				model=self.hp.model.name
				).read()
			read_param_card = SLHA(
				param_card[0],
				self.sampler_id_dir,
				model = self.hp.model.name
				)

		observable_name = self.hp.model.observation.name
		observations = {}
		for obs_name in observable_name:
			if param_card is not None:
				if obs_name in higgs_bounds_results.keys():
					value = float(higgs_bounds_results[obs_name])
				if obs_name in higgs_signals_results.keys():
					value = float(higgs_signals_results[obs_name])
				observations[obs_name] = value
			else:
				observations[obs_name] = None

		params_obs = {**parameters,** observations}
		return params_obs

    def sample_as_numpy(self, parameter_point: np.ndarray) -> np.ndarray:

        params_obs = self.sample(parameter_point)
        
		observable_name = self.hp.model.observation.name
        for obs_name in observable_name:
            print('observable : {} with value: {} '.format(
                obs_name, params_obs[obs_name]
                    ))
        observables_array = np.array(
                [params_obs[obs_name] for obs_name in observable_name]
                )
        return observables_array

    def __call__(self, parameter_point: np.ndarray) -> np.ndarray:
        return self.sample_as_numpy(parameter_point)

	def close(self):
		shutil.rmtree(self.sampler_id_dir)


