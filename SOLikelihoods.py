"""
Likelihood for SZ model

"""


import time
import sys
import numpy as np
import astropy.units as u
from typing import Optional, Sequence, Dict, Any

from Models import TargetData, MapData, HaloModels, FFTs, Projections, Profiles, SHMRs, HODs, Dust, Measurements, Spectra2

from cobaya.yaml import yaml_load_file
from cobaya.theory import Theory

from config import SOLIKET_PATH
sys.path.append(str(SOLIKET_PATH))
from soliket.gaussian import GaussianData, GaussianLikelihood

YAML_FILE = None  # yaml filename set right before running cobaya
VERBOSE = False  # verbose option for testing in notebook
PaperCheck = False  # verbose option for testing in notebook

class SZLikelihood(GaussianLikelihood):
    # Import the dictionary from the yaml file 
    Measurement: Optional[Dict[str, Any]] = None  # Measurement Data
    Profile: Optional[Dict[str, Any]] = None  # Cluster Profile model
    Projection: Optional[Dict[str, Any]] = None  # Projection method
    TargetData: Optional[Dict[str, Any]] = None  # Target sample data (redshift and mass distribution)
    MapData: Optional[Dict[str, Any]] = None  # Map data (beams, responses, etc.)
    HOD: Optional[Dict[str, Any]] = None  # Halo Occupancy Distribution
    HaloModel: Optional[Dict[str, Any]] = None  # Halo Model

    # Anything that should be used for multiple likelihoods should be defined in the initalize
    # Anything that is fixed should also be used here
    def initialize(self):
        if VERBOSE: print("Loading in Fixed Parameters and Data/Model Dictionaries from yaml file")
        # Load the yaml file
        yaml_info = yaml_load_file(YAML_FILE)
        # put shared dicts into each likelihood
        for part in yaml_info['shared']: setattr(self, part, yaml_info['shared'][part])
        # Get all fixed parameters in the params block that have a set value
        self.cosmopars = {k: v["value"] for k, v in yaml_info['params'].items() if isinstance(v, dict) and "value" in v}

        if VERBOSE: print("Loading and Initializing Halo Model:", self.HaloModel['name'], self.HaloModel['spefs'])
        self.halomodel = getattr(HaloModels, self.HaloModel['name'])(self.HaloModel['spefs'] | self.cosmopars)

        if VERBOSE: print("Loading in Measurement Data:", self.Measurement['name'], self.Measurement['spefs'])
        self.meas = getattr(Measurements, self.Measurement['name'])(self.Measurement['spefs'])

        if VERBOSE: print("Loading in Map Data:", self.MapData['name'], self.MapData['spefs'])
        self.mapdata = getattr(MapData, self.MapData['name'])(self.MapData['spefs'])

        if VERBOSE: print("Loading in Target Data:", self.TargetData['name'], self.TargetData['spefs'])
        self.targetdata = getattr(TargetData, self.TargetData['name'])(self.TargetData['spefs'])
        if VERBOSE: print("Making Redshift and Halo Mass Distributions:")
        # Construct redshift distribution
        self.targetdata.make_zdist(**self.TargetData)
        # Construct halo mass distribution, needs halo model for density unit calculations
        self.targetdata.make_Mhdist(halomodel=self.halomodel, **self.TargetData)
        if PaperCheck: self.print_datainfo()

        if VERBOSE: print("Setting up Projection, Beam Convolution, Aperture methods:", self.Projection['name'], self.Projection['spefs'])
        zmean = self.targetdata.zMean if hasattr(self.targetdata, 'zMean') else self.targetdata.distave(self.targetdata.z, self.targetdata.dndz)
        self.proj = getattr(Projections, self.Projection['name'])(Rs=self.meas.R, AngDist=self.halomodel.dA(zmean), **self.Projection['spefs'])
        self.r = np.geomspace(self.proj.r3D.min(), self.proj.r3D.max(), self.Profile['nr'])  # r values for profile evaluation
        
        self.proj2D = self.proj.proj2D(self.r)
        self.beamconvolve = self.proj.beam_convolve(b_ell=self.mapdata.beam_ells, bTF=self.mapdata.beam_data, r_ell=self.mapdata.resp_ells, rTF=self.mapdata.resp_data)
        self.CAP = self.proj.aperture_photometry(f_disc=np.sqrt(2))
        # TODO: this is unnecessarily written up here, try to put more in the projection.py file

        if VERBOSE: print("Setting up Profile Model", self.Profile['name'], self.Profile['spefs'])
        self.profile = getattr(Profiles, self.Profile['name'])(self.Profile['spefs'],rhoc=self.halomodel.rhoc, r200c=self.halomodel.rdel('200c'), dndlogm=self.halomodel.dndlogm, bh=self.halomodel.bh, Plin=self.halomodel.Plin, **self.cosmopars)

        if VERBOSE: print("Setting up Basic Average")
        dndzdlogmhalo_norm = self.targetdata.dndlogMh*self.targetdata.dndz[:, None]/np.trapz(np.trapz(self.targetdata.dndlogMh, self.targetdata.logMh)*self.targetdata.dndz, self.targetdata.z)
        self.ave_smf = lambda prof: np.trapz(np.trapz(prof*dndzdlogmhalo_norm, self.targetdata.logMh), self.targetdata.z)

        if VERBOSE: print("Setting up HOD Average")
        self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])  # Import HOD Model
        # if self.Measurement['spefs']['sample']=='opt':
        #     self.hod.Ncen = lambda logM: lambda p={},  **kwargs: np.ones((logM.size))
        #     self.hod.ncen = lambda ks, zs, logM: lambda p={},  **kwargs: np.ones((ks.size, zs.size, logM.size))
        #     self.hod.Nsat = lambda logM: lambda p={}, **kwargs: np.zeros((logM.size))
        #     self.hod.nsat = lambda ks, zs, logM: lambda p={}, **kwargs: np.zeros((ks.size, zs.size, logM.size))


        self.ave_hod = Spectra2.Popik2026(self.r).HODweighting(self.hod, self.halomodel, self.targetdata)
        self.print_HODinfo()

        if VERBOSE: print("Setting up Data")
        self._get_data()

        if VERBOSE: print("Setting up Profile Model")
        self._init_model()        
        # Full 3D profile to measurement pipeline
        self.project = lambda prof: self.measunit*self.CAP(*self.beamconvolve(self.proj2D(prof)))
        # Full parameters to measurement model
        self.model = lambda params={}: self.project(self.avemeth(self.prof(params)))

        print(f"{self.r.min():.2e}<r<{self.r.max():.2e}, N_r={self.r.shape[0]}")
            
        

    def get_requirements(self):
        return {k: None for k in yaml_load_file(YAML_FILE)['params'].keys()}
    
    def print_datainfo(self):
        print(f"Paper: {getattr(self.targetdata, 'zMin', None)}<z<{getattr(self.targetdata, 'zMax', None)}, zMean={getattr(self.targetdata, 'zMean', None)}")
        print(f"Data: {self.targetdata.z.min():.2f}<z<{self.targetdata.z.max():.2f}, zMean={self.targetdata.distave(self.targetdata.z, self.targetdata.dndz):.2f}")
        print(f"Paper: {getattr(self.targetdata, 'logMhMin', None)}<logM<{getattr(self.targetdata, 'logMhMax', None)}, logMMean={getattr(self.targetdata, 'logMhMean', None)}")
        print(f"Data: {self.targetdata.logMh.min():.2f}<logM<{self.targetdata.logMh.max():.2f}, logMMean={np.log10(self.targetdata.distave(10**self.targetdata.logMh, self.targetdata.dndlogMh, self.targetdata.logMh)):.2f}")
        
    def print_HODinfo(self):
        HODmean = np.log10(Spectra2.Popik2026(self.r).HODweightingM(self.hod, self.halomodel, self.targetdata)(10**self.targetdata.logMh))
        print(f"HOD Model: logMhMean={HODmean:.2f}")
        
    def logp(self, **params_values):
        theory = self._get_theory({**params_values})
        return self.data.loglike(theory)
    
    def test_timing(self):
        timings = {
            "profile1h": [],
            "profile2h": [],
            "profile_total": [],
            "average": [],
            "project": [],
            "forward_total": [],
            "cobaya": [],
            "loglike": [],
        }

        for i in range(10):
            time0 = time.time()

            prof = self.prof({})
            time1 = time.time()
            timings["profile_total"].append((time1 - time0) * 1000)

            profave = self.avemeth(prof)
            time2 = time.time()
            timings["average"].append((time2 - time1) * 1000)

            _ = self.project(profave)
            time3 = time.time()
            timings["project"].append((time3 - time2) * 1000)

            timings["forward_total"].append((time3 - time0) * 1000)

            theory = self._get_theory({})
            time4 = time.time()
            timings["cobaya"].append((time4 - time3) * 1000)

            _ = self.data.loglike(theory)
            time5 = time.time()
            timings["loglike"].append((time5 - time4) * 1000)
            
            prof2h = self.prof2h({})
            time6 = time.time()
            timings["profile2h"].append((time6 - time5) * 1000)
            
            prof1h = self.prof1h({})
            time7 = time.time()
            timings["profile1h"].append((time7 - time6) * 1000)

        results = {}
        for label, values in timings.items():
            mean_t = np.mean(values)
            median_t = np.median(values)
            results[label] = {"mean": mean_t, "median": median_t}
            print(f"{label}: mean={mean_t:.2f} ms, median={median_t:.2f} ms")

        return results



class TSZLikelihood(SZLikelihood):    
    def _get_data(self):
        # Get measurements
        self.data = GaussianData("SZModel", self.meas.R.value, self.meas.tSZ_data.value, self.meas.tSZ_cov.value)
        # Units of measurement
        if self.meas.tSZ_data.unit==u.arcmin**2:
            self.measunit = self.halomodel.pres_to_y(**self.cosmopars)
        elif self.meas.tSZ_data.unit==u.uK*u.arcmin**2:
            self.measunit = self.halomodel.pres_to_uK(self.meas.freq, **self.cosmopars)
            
    def _get_theory(self, params_values):
        return self.model(params_values).value

    def _init_model(self):
        # Profile units
        self.prof = self.profile.Pressure(self.r, self.targetdata.z, self.targetdata.logMh)
        self.prof1h = self.profile.Pressure1h(self.r, self.targetdata.z, self.targetdata.logMh)
        self.prof2h = self.profile.Pressure2h(self.r, self.targetdata.z, self.targetdata.logMh)
        
        # Averaging method
        self.avemeth = self.ave_hod
        # self.aveprof = lambda params: self.ave_smf(self.prof(params))
        self.aveprof = lambda params: self.avemeth(self.prof(params))
        
        # self.aveprof = lambda params: self.profile.Density(self.r, self.targetdata.zMean, self.targetdata.logMhMean)
        
        
class KSZLikelihood(SZLikelihood):    
    def _get_data(self):
        # Get measurements
        self.data = GaussianData("SZModel", self.meas.R.value, self.meas.TkSZ_data.value, self.meas.TkSZ_cov.value)
        # Units of measurement
        self.measunit = self.halomodel.rho_to_TkSZ(**self.cosmopars)
        
    def _get_theory(self, params_values):
        return self.model(params_values).value

    def _init_model(self):
        # Profile units
        self.prof = self.profile.Density(self.r, self.targetdata.z, self.targetdata.logMh)
        self.prof1h = self.profile.Density1h(self.r, self.targetdata.z, self.targetdata.logMh)
        self.prof2h = self.profile.Density2h(self.r, self.targetdata.z, self.targetdata.logMh)
        self.avemeth = self.ave_hod
        # Averaging method
        # self.aveprof = lambda params: self.ave_smf(self.prof(params))
        self.aveprof = lambda params: self.avemeth(self.prof(params))
        # self.aveprof = lambda params: self.profile.Density(self.r, self.targetdata.zMean, self.targetdata.logMhMean)










# class TSZLikelihood(SZLikelihood):    
#     def _get_data(self):
#         # Get measurements
#         self.data = GaussianData("SZModel", self.meas.R, self.meas.y_data, self.meas.y_cov)

#     def _init_model(self):
#                 # Distribution weighted averaging
#         dndzdlogmhalo_norm = self.dndlogmhalo_smf/np.trapz(np.trapz(self.dndlogmhalo_smf, self.logmhalos), self.zs)
#         self.ave_smf = lambda prof: np.trapz(np.trapz(prof*dndzdlogmhalo_norm, self.logmhalos), self.zs)
        
        
        # self.fft = FFTs.mcfit_package(self.rs)
        # self.ks = self.fft.ks

        # if VERBOSE: print("Loading in Halo Occupancy Distribution")  # Load in Map Data
        # self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])  # Import HOD Model
        # self.Nc, self.Ns, self.uck = self.hod.Nc(self.targetdata.logMh), self.hod.Ns(self.targetdata.logMh), self.hod.uck()
        # self.usk = self.hod.usk(self.rs, self.targetdata.z, self.targetdata.logMh, self.r200c, self.fft.FFT3D)


        # self.ave_hod = Spectra.Popik2025().HODweighting(self.Nc, self.Ns, self.uck, self.usk, self.logmhalos, self.zs, self.dndlogmhalo_hmod, self.fft.FFT3D, self.fft.IFFT1D, self.dNdz, **self.cpars)
        
        # self.pth = getattr(Profiles, self.profile['name'])(self.profile['spefs']).Pth(self.rs, self.zs, self.logmhalos, self.rhoc, self.r200c, self.dndlogmhalo_hmod, self.bh, self.ks, self.Plin, self.fft.FFT3D, self.fft.IFFT3D, **self.cpars)
        
        # self.signal = self.proj.Pth_to_y(**self.cpars)
                
        # self.model = lambda params: self.signal(self.ave_smf(self.pth(params)))
        
        










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






    # # Anything that should be used for multiple likelihoods should be defined in the initalize 
    # def initialize(self):
    #     # Get the fixed data and cosmological parameters from the yaml file
    #     yaml_info = yaml_load_file(self.yaml_file)  # Load the yaml file
    #     self.cosmopars = {k: v["value"] for k, v in yaml_info['params'].items() if isinstance(v, dict) and "value" in v}  # get all parameters in the params block that have a set value
    #     for part in yaml_info['shared']: setattr(self, part, yaml_info['shared'][part])  # put spef dictionaries that are shared between likelihoods into the yaml info
                
            
                
                
    #     # # Get basic cosmology functions from astropy
    #     # self.cosmology = astropy.cosmology.Planck18.clone(H0=self.cpars["hh"]*100, Tcmb0=self.cpars["T_CMB"], Om0=self.cpars["Omega_m"], Ob0=self.cpars["Omega_b"])
    #     # self.H = lambda z: (self.cosmology.H(z).to(u.km/u.s/u.Mpc)).value
    #     # self.rhoc = lambda z: (self.cosmology.critical_density(z).to(u.Msun/u.Mpc**3)).value
    #     # self.r200c = lambda z, logmhalo: (10**logmhalo/(4/3*np.pi*200*self.rhoc(z)))**(1/3)
    #     # self.dA = lambda z: self.cosmology.angular_diameter_distance(z).value
        
    #     # # Load Halo Model
    #     # self.halomodel = getattr(HaloModels, self.HaloModel['name'])(self.HaloModel['spefs'])
    #     # self.Plin = lambda k, z: self.halomodel.Plin(k, z, **self.cpars)
    #     # self.bh = lambda z, logmhalo: self.halomodel.bh(z, logmhalo, **self.cpars)
    #     # self.dndlogmhalo_hmod = lambda z, logmhalo: self.halomodel.HMF(z, logmhalo, **self.cpars)
        
    #     # Set arrays limits
    #     self.halodist = getattr(SMFs, self.SMF['name'])(self.SMF['spefs'])
    #     self.zs = self.halodist.z
    #     self.logmhalos = self.halodist.logmhalo
    #     self.dndlogmhalo_smf = self.halodist.dndlogmhalo(**self.cpars)
    #     self.dNdz = self.halodist.dNdz(**self.cpars)
    #     self.zave = np.trapz(self.zs*self.dNdz, self.zs)/np.trapz(self.dNdz, self.zs)

    #     # Distribution weighted averaging
    #     dndzdlogmhalo_norm = self.dndlogmhalo_smf/np.trapz(np.trapz(self.dndlogmhalo_smf, self.logmhalos), self.zs)
    #     self.ave_smf = lambda prof: np.trapz(np.trapz(prof*dndzdlogmhalo_norm, self.logmhalos), self.zs)

    #     # Setup projection and get r values
    #     self.meas = getattr(Data, self.measurement['name'])(self.measurement['spefs'])  # Import Measurement
    #     self.proj = Projections.Popik2025(self.meas.R, self.dA(self.zave), self.meas.beam_ells, self.meas.beam_data, self.meas.resp_ells, self.meas.resp_data)
    #     self.rs = self.proj.rs
    #     self.fft = FFTs.mcfit_package(self.rs)
    #     self.ks = self.fft.ks

    #     # HOD averaged
    #     self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])  # Import HOD Model
    #     self.Nc = self.hod.Nc(self.logmhalos)
    #     self.Ns = self.hod.Ns(self.logmhalos)
    #     self.uck = self.hod.uck()
    #     self.usk = self.hod.usk(self.rs, self.zs, self.logmhalos, self.r200c, self.fft.FFT3D)
    #     self.ave_hod = Spectra.Popik2025().HODweighting(self.Nc, self.Ns, self.uck, self.usk, self.logmhalos, self.zs, self.dndlogmhalo_hmod, self.fft.FFT3D, self.fft.IFFT1D, self.dNdz, **self.cpars)



    #     self._init_model()        
    #     self._get_data()
                 
    # def logp(self, **params_values):
    #     theory = self._get_theory({**params_values})
    #     return self.data.loglike(theory)

    # def get_requirements(self):
    #     return {k: None for k in yaml_load_file(self.yaml_file)['params'].keys()}
    
    # def _get_theory(self, params_values):
    #     return self.model(params_values)