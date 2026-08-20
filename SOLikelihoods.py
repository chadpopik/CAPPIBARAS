"""
Likelihood for SZ model
"""

from config import *
from typing import Optional, Sequence, Dict, Any
from cobaya.yaml import yaml_load_file
sys.path.append(str(SOLIKET_PATH))
from soliket.gaussian import GaussianData, GaussianLikelihood

from ForwardModel import SZObservable


class SOLikelihood(GaussianLikelihood, SZObservable):
    VERBOSE: bool = False
    PAPERCHECK: bool = False
    YAML_FILE: str | None = None

    for _name in SZObservable.submodels:
        locals()[_name] = None
        __annotations__[_name] = Optional[Dict[str, Any]]
    del _name

    def initialize(self):
        # Load the yaml file and incorperate everything in the "shared" block into the likelihood
        if self.VERBOSE: print("Loading in Fixed Parameters and Data/Model Dictionaries from yaml file")
        yaml_info = yaml_load_file(self.YAML_FILE)
        for part in yaml_info['shared']: setattr(self, part, yaml_info['shared'][part])
        
        # Load up the forward model
        Models = {k: getattr(self, k) for k in SZObservable.submodels if getattr(self, k) is not None}
        SZObservable.__init__(self, VERBOSE=self.VERBOSE, PAPERCHECK=self.PAPERCHECK, Models=Models)
        
        self.model = self.forward_model
        
        self._get_data()
        
    def get_requirements(self):
        return {k: None for k in yaml_load_file(self.YAML_FILE)['params'].keys()}
        
    def _get_theory(self, params_values):
        return self.model(params_values).value
    
    def _get_data(self):
        self.data = GaussianData("SZModel", self.R.value, self.meas_data.value, self.meas_cov.value)
        
    def logp(self, **params_values):
        theory = self._get_theory({**params_values})
        return self.data.loglike(theory) 