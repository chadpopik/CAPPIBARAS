"""
Likelihood for SZ model

- TODO 1: Is there a way to not way to write a preinit line for each value that's needed?
- TODO 2: Is there a way to just know the yaml file and more easily load the original params?
- TODO 3: Should I explore with actually creating a theory class to calculate some of the HOD/HMF type things?
- TODO 4: How to decide rs array? 
"""


import sys
import numpy as np
import astropy
import astropy.units as u
import astropy.constants as c
from typing import Optional, Sequence, Dict, Any
from astropy.cosmology import default_cosmology, Planck18
from scipy.interpolate import RegularGridInterpolator

from Models import Data, HaloModels, FFTs, Spectra, Projections, Profiles, SHMRs, HODs, SMFs, Dust

from cobaya.yaml import yaml_load_file

sys.path.append('/global/homes/c/cpopik/soliket/')
from soliket.gaussian import GaussianData, GaussianLikelihood


class SZLikelihood(GaussianLikelihood):
    # TODO 1
    measurement: Optional[Dict[str, Any]] = None
    onehalo: Optional[Dict[str, Any]] = None
    twohalo: Optional[Dict[str, Any]] = None
    DustModel: Optional[Dict[str, Any]] = None
    
    SMF: Optional[Dict[str, Any]] = None
    HOD: Optional[Dict[str, Any]] = None
    SHMR: Optional[Dict[str, Any]] = None
    mass_function: Optional[Dict[str, Any]] = None

    # TODO 2
    yaml_file = "/global/homes/c/cpopik/CAPPIBARAS/runchains.yaml"

    # Anything that should be used for multiple likelihoods should be defined in the initalize 
    def initialize(self):  # TODO 3
        # Fetch the cosmology parameters from the yaml file
        self.cpars = {k: v["value"] for k, v in yaml_load_file(self.yaml_file)['params'].items() if isinstance(v, dict) and "value" in v}  
        
        print("Loading Modules")
        self.cosmology = astropy.cosmology.LambdaCDM(H0=self.cpars["hh"]*100, Tcmb0=2.726, Om0=self.cpars["Omega_m"], Ode0=self.cpars["Omega_L"], Ob0=self.cpars["Omega_b"])
        self.shmr = getattr(SHMRs, self.SHMR['name'])(self.SHMR['spefs'])
        self.halomodel = getattr(HaloModels, self.mass_function['name'])(self.mass_function['spefs'])
        self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])
        self.smf = getattr(SMFs, self.SMF['name'])(self.SMF['spefs'])

        print("Loading Data")
        self._get_data()
        
        print("Pre-calculating cosmology values")  # Get basic cosmology functions from astropy
        self.fft = FFTs.mcfit_package(self.rs)
        self.ks = self.fft.ks
        self.Hs = (self.cosmology.H(self.zs).to(u.km/u.s/u.Mpc)).value
        self.rhocs = (self.cosmology.critical_density(self.zs).to(u.Msun/u.Mpc**3)).value
        self.r200cs = (10**self.logmhalos/(4/3*np.pi*200*self.rhocs[:, None]))**(1/3)
        self.Plins = self.halomodel.Plin(self.ks, self.zs, **self.cpars)
        self.bhs = self.halomodel.bh(self.zs, self.logmhalos, **self.cpars)
        self.hmf_hmod = self.halomodel.HMF(self.zs, self.logmhalos, **self.cpars)
        self.zave = np.trapz(self.zs*self.dNdz, self.zs)/np.trapz(self.dNdz, self.zs)
        self.dA_ave = self.cosmology.angular_diameter_distance(self.zave).value
        
        print("Setting up Foward Model")
        # HOD averaged
        self.Nc, self.Ns = self.hod.Nc(self.logmhalos), self.hod.Ns(self.logmhalos)
        self.uck, self.usk = self.hod.uck(), self.hod.usk(self.rs, self.r200cs, self.fft.FFT3D)
        self.ave_hod = Spectra.HODweighting(self.Nc, self.Ns, self.uck, self.usk, self.logmhalos, self.zs, self.hmf_hmod, self.fft.FFT3D, self.fft.IFFT1D, self.dNdz, **self.cpars)
        
        # SMF averaging
        dndzdlogmhalo = self.hmf_smf/(self.zs[1]-self.zs[0])
        dndzdlogmhalo_norm = dndzdlogmhalo/np.trapz(np.trapz(dndzdlogmhalo, self.logmhalos), self.zs)
        self.ave_smf = lambda prof: np.trapz(np.trapz(prof*dndzdlogmhalo_norm, self.logmhalos), self.zs)
         
        self._init_model()        
        
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
        self.meas = getattr(Data, self.measurement['name'])(self.measurement['spefs'])
        self.data = GaussianData("SZModel", self.meas.thetas, self.meas.tSZdata, self.meas.tSZcov)
        
        # Get the HMF from the SMF using a SHMR
        self.smf.make_SMF(**self.cpars)
        dndlogmhalo_smf = self.smf.hmf_from_smf(lambda mstar: self.shmr.SHMR(mstar)(), **self.cpars)
        
        # Set arrays limits
        self.rs = np.logspace(-1.5, 1.5, 100)  # TODO 4
        zbins = np.linspace(0.3, 0.6, 11)
        logmhalobins = np.linspace(11, 15, 51)
        self.logmhalos, self.zs = (logmhalobins[1:]+logmhalobins[:-1])/2, (zbins[1:]+zbins[:-1])/2
        
        # Interpolate the HMF onto the new grid
        intp_points = np.stack(np.meshgrid(self.zs, self.logmhalos, indexing='ij'), axis=-1)
        self.hmf_smf = RegularGridInterpolator((self.smf.z, self.smf.logmhalo), dndlogmhalo_smf*[self.smf.logmhalo[1]-self.smf.logmhalo[0]], bounds_error=False, fill_value=np.nan)(intp_points)/(self.logmhalos[1]-self.logmhalos[0])

        # Get the redshift distribution and also interpolate
        dNdz_z = np.trapz(self.smf.dndlogmstar(**self.cpars), self.smf.logmstar)*self.smf.volumes(**self.cpars)
        self.dNdz = np.interp(self.zs, self.smf.z, dNdz_z)/(self.zs[1]-self.zs[0])

    def _init_model(self):
        self.pth_1h = getattr(Profiles, self.onehalo['name'])(self.onehalo['spefs']).Pth1h(self.rs, self.zs, self.logmhalos, self.rhocs, self.r200cs, **self.cpars)
        self.pth_2h = getattr(Profiles, self.twohalo['name'])(self.twohalo['spefs']).Pth2h(self.rs, self.zs, self.logmhalos, self.rhocs, self.r200cs, self.Plins, self.bhs, self.hmf_smf, self.fft.FFT3D, self.fft.IFFT3D, self.ks, **self.cpars)
        self.pth = lambda params={}: self.pth_1h(params) + self.pth_2h(params)[..., None]
        
        self.tSZ_uK = Projections.Pth_to_uK(self.rs, self.meas.thetas, self.dA_ave, self.meas.beam_data, self.meas.beam_ells, self.meas.resp_data, self.meas.resp_ells, **self.cpars)
        
        self.dustprof = getattr(Dust, self.DustModel['name'])(self.DustModel['spefs']).dust_uKarcmin(self.meas.thetas, 150, **self.cpars)()
        
        self.model = lambda params: self.tSZ_uK(self.ave_smf(self.pth(params))) + self.dustprof
        

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

class CggLikelihood(SZLikelihood):
    def _get_data(self):
        pass
    
    def _init_model(self):
        pass