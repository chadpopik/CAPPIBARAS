"""
Likelihood for SZ model

"""


import sys
import numpy as np
import astropy
import astropy.units as u
import astropy.constants as c
from typing import Optional, Sequence, Dict, Any
from astropy.cosmology import default_cosmology, Planck18
from scipy.interpolate import RegularGridInterpolator

from CAPPIBARAS.Testing import SMFs
from Models import Data, HaloModels, FFTs, Spectra, Projections, Profiles, SHMRs, HODs, Dust

from cobaya.yaml import yaml_load_file

sys.path.append('/global/homes/c/cpopik/soliket/')
from soliket.gaussian import GaussianData, GaussianLikelihood



class SZLikelihood(GaussianLikelihood):
    measurement: Optional[Dict[str, Any]] = None
    profile: Optional[Dict[str, Any]] = None
    SMF: Optional[Dict[str, Any]] = None
    HOD: Optional[Dict[str, Any]] = None
    HaloModel: Optional[Dict[str, Any]] = None
    yaml_file: Optional[str] = None

    # Anything that should be used for multiple likelihoods should be defined in the initalize 
    def initialize(self):
        yaml_info = yaml_load_file(self.yaml_file)
        self.cpars = {k: v["value"] for k, v in yaml_info['params'].items() if isinstance(v, dict) and "value" in v}
        for part in yaml_info['shared']:
            setattr(self, part, yaml_info['shared'][part])
                
        # Get basic cosmology functions from astropy
        self.cosmology = astropy.cosmology.Planck18.clone(H0=self.cpars["hh"]*100, Tcmb0=self.cpars["T_CMB"], Om0=self.cpars["Omega_m"], Ob0=self.cpars["Omega_b"])
        self.H = lambda z: (self.cosmology.H(z).to(u.km/u.s/u.Mpc)).value
        self.rhoc = lambda z: (self.cosmology.critical_density(z).to(u.Msun/u.Mpc**3)).value
        self.r200c = lambda z, logmhalo: (10**logmhalo/(4/3*np.pi*200*self.rhoc(z)))**(1/3)
        self.dA = lambda z: self.cosmology.angular_diameter_distance(z).value
        
        # Load Halo Model
        self.halomodel = getattr(HaloModels, self.HaloModel['name'])(self.HaloModel['spefs'])
        self.Plin = lambda k, z: self.halomodel.Plin(k, z, **self.cpars)
        self.bh = lambda z, logmhalo: self.halomodel.bh(z, logmhalo, **self.cpars)
        self.dndlogmhalo_hmod = lambda z, logmhalo: self.halomodel.HMF(z, logmhalo, **self.cpars)
        
        # Set arrays limits
        self.halodist = getattr(SMFs, self.SMF['name'])(self.SMF['spefs'])
        self.zs = self.halodist.z
        self.logmhalos = self.halodist.logmhalo
        self.dndlogmhalo_smf = self.halodist.dndlogmhalo(**self.cpars)
        self.dNdz = self.halodist.dNdz(**self.cpars)
        self.zave = np.trapz(self.zs*self.dNdz, self.zs)/np.trapz(self.dNdz, self.zs)

        # Distribution weighted averaging
        dndzdlogmhalo_norm = self.dndlogmhalo_smf/np.trapz(np.trapz(self.dndlogmhalo_smf, self.logmhalos), self.zs)
        self.ave_smf = lambda prof: np.trapz(np.trapz(prof*dndzdlogmhalo_norm, self.logmhalos), self.zs)

        # Setup projection and get r values
        self.meas = getattr(Data, self.measurement['name'])(self.measurement['spefs'])  # Import Measurement
        self.proj = Projections.Popik2025(self.meas.R, self.dA(self.zave), self.meas.beam_ells, self.meas.beam_data, self.meas.resp_ells, self.meas.resp_data)
        self.rs = self.proj.rs
        self.fft = FFTs.mcfit_package(self.rs)
        self.ks = self.fft.ks

        # HOD averaged
        self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])  # Import HOD Model
        self.Nc = self.hod.Nc(self.logmhalos)
        self.Ns = self.hod.Ns(self.logmhalos)
        self.uck = self.hod.uck()
        self.usk = self.hod.usk(self.rs, self.zs, self.logmhalos, self.r200c, self.fft.FFT3D)
        self.ave_hod = Spectra.Popik2025().HODweighting(self.Nc, self.Ns, self.uck, self.usk, self.logmhalos, self.zs, self.dndlogmhalo_hmod, self.fft.FFT3D, self.fft.IFFT1D, self.dNdz, **self.cpars)



        self._init_model()        
        self._get_data()
                 
    def logp(self, **params_values):
        theory = self._get_theory({**params_values})
        return self.data.loglike(theory)

    def get_requirements(self):
        return {k: None for k in yaml_load_file(self.yaml_file)['params'].keys()}
    
    def _get_theory(self, params_values):
        return self.model(params_values)


class TSZLikelihood(SZLikelihood):    
    def _get_data(self):
        # Get measurements
        self.data = GaussianData("SZModel", self.meas.R, self.meas.y_data, self.meas.y_cov)

    def _init_model(self):
        self.pth = getattr(Profiles, self.profile['name'])(self.profile['spefs']).Pth(self.rs, self.zs, self.logmhalos, self.rhoc, self.r200c, self.dndlogmhalo_hmod, self.bh, self.ks, self.Plin, self.fft.FFT3D, self.fft.IFFT3D, **self.cpars)
        
        self.signal = self.proj.Pth_to_y(**self.cpars)
                
        self.model = lambda params: self.signal(self.ave_smf(self.pth(params)))


# class KSZLikelihood(SZLikelihood):
#     # Data is specific to measurement, so describe in the topmost likelihood
#     def _get_data(self):
#         self.meas = getattr(Data, self.DataUse['name'])(self.DataUse['spefs'])
        
#         self.data = GaussianData("SZModel", self.meas.thetas, self.meas.kSZdata, self.meas.kSZcov)

#     def _init_model(self):
#         self.rho_1h = getattr(Profiles, self.onehalo['name'])(self.onehalo['spefs']).rho1h(self.rs, self.zs, self.mhalos, self.rhoc_func, self.r200c_func, **self.cpars)
#         self.rho_2h = getattr(Profiles, self.twohalo['name'])(self.twohalo['spefs']).rho2h(self.rs, self.zs, self.mhalos)
#         self.prof = lambda params={}: self.rho_1h(params) + self.rho_2h(params)
        
#         self.sign = ForwardModel.rho_to_muK(**self.cpars)
        
#         # print(f"Density 1h: {self.ave_SMF(self.prof())[0:20]}, 2halo: {self.rho_2h()[0:20, 0, 0]}")
#         # print(f"Theory: {self._get_theory({})}")
#         # print(f"Data:{self.meas.kSZdata}")

#     def _get_theory(self, params_values):
#         rhos = self.prof(params_values)
#         rho_ave = self.ave_SMF(rhos)  # Can switch this out with the HOD version as desired
#         sig = self.proj(rho_ave)
#         return self.sign(sig)

# class CggLikelihood(SZLikelihood):
#     def _get_data(self):
#         pass
    
#     def _init_model(self):
#         pass