"""
Likelihood for SZ model

TODO: Is there a way to not way to write a preinit line for each value that's needed?
TODO: Is there a way to just know the yaml file and more easily load the original params?
TODO: Should I explore with actually creating a theory class to calculate some of the HOD/HMF type things?
TODO: How to decide rs array? And mass and redshift limits?
"""

from Basics import *
import ForwardModel, Measurements, Profiles, SHMRs, HODs, SMFs, HMFs, FFTs

from typing import Optional, Sequence, Dict, Any
from cobaya.yaml import yaml_load_file

sys.path.append('/global/homes/c/cpopik/Git/SOLikeT')
from soliket.gaussian import GaussianData, GaussianLikelihood


class SZLikelihoodTEST(GaussianLikelihood):
    DataUse: Optional[Dict[str, Any]] = None
    galaxy_distribution: Optional[Dict[str, Any]] = None
    HOD: Optional[Dict[str, Any]] = None
    SHMR: Optional[Dict[str, Any]] = None
    onehalo: Optional[Dict[str, Any]] = None
    twohalo: Optional[Dict[str, Any]] = None
    mass_function: Optional[Dict[str, Any]] = None
    FFTtype: Optional[Dict[str, Any]] = None
    thingtest: Optional[Dict[str, Any]] = None
    
    yaml_file = "/global/homes/c/cpopik/Git/Capybara/cobayatest_chains.yaml"

    # Anything that should be used for multiple likelihoods should be defind in the initalize 
    def initialize(self):
        
        print(thingtest['name'])
        # Retrieve fixed cosmological parameters from the yaml file to set up cosmological functions, and original guesses for HOD/Profile parameters for testing purposes
        self.fixedpar = {k: v["value"] for k, v in yaml_load_file(self.yaml_file)['params'].items() if isinstance(v, dict) and "value" in v}
        # self.varpar =  {k: (v['prior']['max']+v['prior']['max'])/2 for k, v in yaml_load_file(self.yaml_file)['params'].items() if isinstance(v, dict) and "prior" in v}
        self.varpar = {k: v["ref"] for k, v in yaml_load_file(self.yaml_file)['params'].items() if isinstance(v, dict) and "ref" in v}
        self.initpar = self.fixedpar | self.varpar
        
        self._get_data()
        self.data = GaussianData("SZModel", self.thetas, self.meas, self.cov)

        print("Loading Cosmology functions") # The following functions can be from anywhere, but the model requires them. I just use astropy cosmology functions here
        self.cosmology = astropy.cosmology.LambdaCDM(H0=self.initpar["hh"]*100, Tcmb0=2.726, Om0=self.initpar["Omega_m"], Ode0=self.initpar["Omega_L"], Ob0=self.initpar["Omega_b"])
        self.H_func = lambda z: self.cosmology.H(z).to(u.km/u.s/u.Mpc).value
        self.rhoc_func = lambda z: self.cosmology.critical_density(z).to(u.Msun/u.Mpc**3).value
        self.dA_func = lambda z: self.cosmology.angular_diameter_distance(z).value
        self.r200c_func = lambda m200c, z: (m200c/(4/3*np.pi*200*self.rhoc_func(z)))**(1/3)

        # Get stellar mass and redshift distributions, which also give m200cs and zs arrays/bounds, and calculate an average redshift. If using HOD, you can just define your own m/z arrays and average z.
        self.galaxydist = SMFs.get_Class(self.galaxy_distribution['name'])(*self.galaxy_distribution['spefs'])
        self.dndmdz_SMF, self.mstars, self.zs = self.galaxydist.SMF, self.galaxydist.mstars, self.galaxydist.zs
        
        self.dndmdz_SMF = self.dndmdz_SMF[:, (self.zs>=0.4) & (self.zs<=0.7)]
        self.zs = self.zs[(self.zs>=0.4) & (self.zs<=0.7)]
        
        self.ave_z = np.sum(np.trapz(self.dndmdz_SMF, self.mstars, axis=0)*self.zs)/np.sum(np.trapz(self.dndmdz_SMF, self.mstars, axis=0))

        # Use a SHMR to convert stellar masses to halo masses
        self.m200cs = SHMRs.get_Class(self.SHMR['name'])(*self.SHMR['spefs']).SHMR(self.mstars)

        # TODO: decide how to add mass cuts beyond what the distribution asks
        self.dndmdz_SMF = self.dndmdz_SMF[(self.m200cs>10**12) & (self.m200cs<10**14), :]
        self.m200cs = self.m200cs[(self.m200cs>10**12) & (self.m200cs<10**14)]
        
        # Create a HMF function to use with the HOD
        self.hmf = HMFs.get_Class(self.mass_function['name'])(*self.mass_function['spefs']).HMF(self.m200cs, self.zs)

        # Define an HOD Model used for this method
        self.HOD = HODs.get_Class(self.HOD['name'])(*self.HOD['spefs'])

        # TODO: find better method to decide which rs to use
        self.rs = np.logspace(-1, 1, 100)

        # The HOD method requires a method of FFT back and forth
        self.ks, self.FFT_func= FFTs.get_Class(self.FFTtype['name'])(self.rs).FFT3D()
        self.rs_rev, self.IFFT_func= FFTs.get_Class(self.FFTtype['name'])(self.rs).IFFT1D()
        
        self.ave_SMF = ForwardModel.weighting(self.m200cs, self.zs, self.dndmdz_SMF)
        self.ave_HOD = ForwardModel.HODweighting(self.m200cs, self.zs, self.rs, self.HOD, self.hmf, self.r200c_func, self.H_func, self.FFT_func, self.IFFT_func)
        
        self.proj = ForwardModel.project_Hankel(self.rs, self.thetas, self.dA_func(self.ave_z), self.beam_data, self.beam_ells, self.resp_data, self.resp_ells)

        print("Initialize Model")  # 
        self._init_model()

    def logp(self, **params_values):
        theory = self._get_theory({**params_values})
        return self.data.loglike(theory)


class TSZLikelihoodTEST(SZLikelihoodTEST):  # this is for GNFW model
    def get_requirements(self):
        return {k: None for k in self.initpar.keys()}
    
    # Data is specific to measurement, so describe in the topmost likelihood
    def _get_data(self):
        datastudy = Measurements.get_Class(self.DataUse['name'])(*self.DataUse['spefs'])
        
        self.thetas, self.meas, self.cov = datastudy.thetas, datastudy.tSZdata, datastudy.tSZcov
        self.freq = datastudy.freq
        self.dust = datastudy.dustprof
        self.beam_ells, self.beam_data = datastudy.beam_ells, datastudy.beam_data
        self.resp_ells, self.resp_data = datastudy.resp_ells, datastudy.resp_data

    def _init_model(self):
        self.pth_1h = Profiles.get_Class(self.onehalo['name'])(*self.onehalo['spefs']).Pth1h(self.rs, self.m200cs, self.zs, self.rhoc_func, self.r200c_func, self.initpar)
        self.pth_2h = Profiles.get_Class(self.twohalo['name'])(*self.twohalo['spefs']).Pth2h(self.rs, self.m200cs, self.zs)
        self.prof = lambda params: self.pth_1h(params) + self.pth_2h(params)
        
        self.sign = ForwardModel.Pth_to_muK(freq=self.freq, **self.initpar)
        
        print(self.meas)
        
        print(f"Density: 1halo: {self.pth_1h(self.initpar)[0:20, 5, 2]}, 2halo: {self.pth_2h(self.initpar)[0:20, 5, 2]}")
        print(f"Averaged Density: {self.ave_SMF(self.prof(self.initpar))[0:20]}")
        print(f"Theory: {self._get_theory(self.initpar)}")

    def _get_theory(self, params_values):
        pths = self.prof(params_values)
        pth_ave = self.ave_SMF(pths)  # Can switch this out with the HOD version as desired
        sig = self.proj(pth_ave)
        return self.sign(sig) + self.dust
    

class KSZLikelihoodTEST(SZLikelihoodTEST):  # this is for GNFW model
    def get_requirements(self):
        return {k: None for k in self.initpar.keys()}

    # Data is specific to measurement, so describe in the topmost likelihood
    def _get_data(self):
        datastudy = Measurements.get_Class(self.DataUse['name'])(*self.DataUse['spefs'])
        
        self.thetas, self.meas, self.cov = datastudy.thetas, datastudy.kSZdata, datastudy.kSZcov
        self.freq = datastudy.freq
        self.dust = datastudy.dustprof
        self.beam_ells, self.beam_data = datastudy.beam_ells, datastudy.beam_data
        self.resp_ells, self.resp_data = datastudy.resp_ells, datastudy.resp_data

    def _init_model(self):
        self.rho_1h = Profiles.get_Class(self.onehalo['name'])(*self.onehalo['spefs']).rho1h(self.rs, self.m200cs, self.zs, self.rhoc_func, self.r200c_func, self.initpar)
        self.rho_2h = Profiles.get_Class(self.onehalo['name'])(*self.onehalo['spefs']).rho2h(self.rs, self.m200cs, self.zs)

        self.prof = lambda params: self.rho_1h(params)+self.rho_2h(params)
        
        self.sign = ForwardModel.rho_to_muK(freq=self.freq, **self.initpar)
        
        print(self.meas)
        
        print(f"Density: 1halo: {self.rho_1h(self.initpar)[0:20, 5, 2]*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)}, 2halo: {self.rho_2h()[0:20, 5, 2]}")
        print(f"Averaged Density: {self.ave_SMF(self.prof())[0:20]}")
        print(f"Theory: {self._get_theory()}")


    def _get_theory(self, params_values):
        pths = self.prof(params_values)
        pth_ave = self.ave_SMF(pths)  # Can switch this out with the HOD version as desired
        sig = self.proj(pth_ave)
        return self.sign(sig)