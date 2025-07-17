"""
Likelihood for SZ model
"""

from Basics import *
from Models import ForwardModel, Measurements, Profiles, SHMRs, HODs, SMFs, HMFs

from cobaya.yaml import yaml_load_file

sys.path.append('/global/homes/c/cpopik/SOLikeT')
from soliket.gaussian import GaussianData, GaussianLikelihood


class SZLikelihood(GaussianLikelihood):
    # TODO 1
    DataUse: Optional[Dict[str, Any]] = None
    onehalo: Optional[Dict[str, Any]] = None
    twohalo: Optional[Dict[str, Any]] = None
    
    galaxy_distribution: Optional[Dict[str, Any]] = None
    HOD: Optional[Dict[str, Any]] = None
    SHMR: Optional[Dict[str, Any]] = None
    mass_function: Optional[Dict[str, Any]] = None

    yaml_file = "/global/homes/c/cpopik/Capybara/runchains.yaml"

    # Anything that should be used for multiple likelihoods should be defind in the initalize 
    def initialize(self):
        self.cpars = {k: v["value"] for k, v in yaml_load_file(self.yaml_file)['params'].items() if isinstance(v, dict) and "value" in v}
        
        print("Loading Data")
        self._get_data()   

        print("Loading Cosmology functions")
        self.cosmology = astropy.cosmology.LambdaCDM(H0=self.cpars["hh"]*100, Tcmb0=2.726, Om0=self.cpars["Omega_m"], Ode0=self.cpars["Omega_L"], Ob0=self.cpars["Omega_b"])
        self.H_func = lambda z: self.cosmology.H(z).to(u.km/u.s/u.Mpc).value
        self.rhoc_func = lambda z: self.cosmology.critical_density(z).to(u.Msun/u.Mpc**3).value
        self.dA_func = lambda z: self.cosmology.angular_diameter_distance(z).value
        self.r200c_func = lambda z, m200c: (m200c/(4/3*np.pi*200*self.rhoc_func(z)))**(1/3)

        print("Loading Galaxy Distributions")
        self.smf = getattr(SMFs, self.galaxy_distribution['name'])(self.galaxy_distribution['spefs'])
        self.gdist = self.smf.gdist(**self.cpars)
        self.mstars, self.zs = self.smf.mstars, self.smf.zs

        print("Loading SHMR")
        self.shmr = getattr(SHMRs, self.SHMR['name'])(self.SHMR['spefs'])
        self.mhalos = self.shmr.SHMR(self.mstars)
        
        # Cut down the mrange
        mrange = self.mhalos < self.meas.mhalomax
        self.gdist, self.mstars, self.mhalos = self.gdist[:, mrange], self.mstars[mrange], self.mhalos[mrange]
        self.mhaloave = self.shmr.SHMR(np.sum(self.gdist*self.mstars)/np.sum(self.gdist))
        self.zave = np.sum(self.gdist*self.zs[:, None])/np.sum(self.gdist)
        
        print("Loading HMF")
        self.halomodel = getattr(HMFs, self.mass_function['name'])(self.mass_function['spefs'])
        self.hmf = self.halomodel.HMF(self.zs, self.mhalos, **self.cpars)
        
        print("Loading HOD")
        self.hod = getattr(HODs, self.HOD['name'])(self.HOD['spefs'])

        self.rs = np.logspace(-1, 1, 100)

        print("Loading Average Functions")
        self.ave_SMF = ForwardModel.weighting(self.gdist)
        self.ave_HOD = ForwardModel.HODweighting(self.rs, self.zs, self.mhalos, self.hod.Nc, self.hod.Ns, self.hod.uSat, self.hmf, self.r200c_func, self.H_func, **self.cpars)

        print("Loading Projection Functions")
        self.proj = ForwardModel.project_Hankel(self.rs, self.meas.thetas, self.dA_func(self.smf.zave()), self.meas.beam_data, self.meas.beam_ells, self.meas.resp_data, self.meas.resp_ells)

        print("Initializing Model")
        self._init_model()

    def logp(self, **params_values):
        theory = self._get_theory({**params_values})
        return self.data.loglike(theory)

    def get_requirements(self):
        return {k: None for k in yaml_load_file(self.yaml_file)['params'].keys()}


class TSZLikelihood(SZLikelihood):    
    def _get_data(self):
        self.meas = getattr(Measurements, self.DataUse['name'])(self.DataUse['spefs'])
        
        self.data = GaussianData("SZModel", self.meas.thetas, self.meas.tSZdata, self.meas.tSZcov)

    def _init_model(self):
        self.pth_1h = getattr(Profiles, self.onehalo['name'])(self.onehalo['spefs']).Pth1h(self.rs, self.zs, self.mhalos, self.rhoc_func, self.r200c_func, **self.cpars)
        
        self.pth_2h = getattr(Profiles, self.twohalo['name'])(self.twohalo['spefs']).Pth2h(self.rs, np.array([self.zave]), self.mhalos, self.rhoc_func, self.r200c_func, self.halomodel.Plin, self.halomodel.bh, self.halomodel.HMF, **self.cpars)
        
        self.prof = lambda params={}: self.pth_1h(params) + self.pth_2h(params)
        
        self.sign = ForwardModel.Pth_to_muK(freq=self.meas.freq, **self.cpars)
        
        # print(f"Pressure 1h: {self.ave_SMF(self.prof())[0:20]}, 2halo: {self.pth_2h()[0:20, 0, 0]}")
        # print(f"Theory: {self._get_theory({})}")
        # print(f"Data:{self.meas.tSZdata}")

    def _get_theory(self, params_values):
        pths = self.prof(params_values)
        pth_ave = self.ave_SMF(pths)  # Can switch this out with the HOD version as desired
        sig = self.proj(pth_ave)
        return self.sign(sig) + self.meas.dustprof


class KSZLikelihood(SZLikelihood):
    # Data is specific to measurement, so describe in the topmost likelihood
    def _get_data(self):
        self.meas = getattr(Measurements, self.DataUse['name'])(self.DataUse['spefs'])
        
        self.data = GaussianData("SZModel", self.meas.thetas, self.meas.kSZdata, self.meas.kSZcov)

    def _init_model(self):
        self.rho_1h = getattr(Profiles, self.onehalo['name'])(self.onehalo['spefs']).rho1h(self.rs, self.zs, self.mhalos, self.rhoc_func, self.r200c_func, **self.cpars)
        self.rho_2h = getattr(Profiles, self.twohalo['name'])(self.twohalo['spefs']).rho2h(self.rs, self.zs, self.mhalos)
        self.prof = lambda params={}: self.rho_1h(params) + self.rho_2h(params)
        
        self.sign = ForwardModel.rho_to_muK(**self.cpars)
        
        # print(f"Density 1h: {self.ave_SMF(self.prof())[0:20]}, 2halo: {self.rho_2h()[0:20, 0, 0]}")
        # print(f"Theory: {self._get_theory({})}")
        # print(f"Data:{self.meas.kSZdata}")

    def _get_theory(self, params_values):
        rhos = self.prof(params_values)
        rho_ave = self.ave_SMF(rhos)  # Can switch this out with the HOD version as desired
        sig = self.proj(rho_ave)
        return self.sign(sig)