"""
Collection of measurements/covariances from different stacking/cross-correlation studies and corresponding components needed for forward modelling and inference, such as the beam, response, frequency, and dust model. 
Classes should have consistently titled attributes and units.
"""

from Basics import *


def fnu(nu, T_cmb):
    x = (c.h * nu*u.GHz / (c.k_B * T_cmb*u.K)).decompose().value
    ans = x / np.tanh(x / 2.0) - 4.0
    return ans

class BaseData:
    def checkspefs(self, spefs, required):  # Check if the spef is in the list
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s").keys():
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s').keys()}")
            else:
                setattr(self, mname, getattr(self, f'{mname}s')[spefs[mname]])
                

# TODO 1
roughdustx, roughdusty = np.array([1.64, 1.83, 2.01, 2.17, 2.34, 2.54, 2.71, 2.9 , 3.1 , 3.27, 3.49, 3.49, 3.66, 3.85, 4.06, 4.06, 4.29, 4.45, 4.45, 4.64, 4.64, 4.83, 4.83, 5.02, 5.02, 5.2 , 5.2 , 5.39, 5.58, 5.77, 5.77, 5.93, 5.93]), np.array([1.05, 1.23, 1.31, 1.43, 1.58, 1.65, 1.77, 1.75, 1.8 , 1.85, 1.85, 1.85, 1.75, 1.69, 1.54, 1.54, 1.45, 1.31, 1.31, 1.11, 1.11, 0.97, 0.97, 0.84, 0.84, 0.73, 0.73, 0.57, 0.46, 0.34, 0.34, 0.15, 0.15])

class Hadzhiyska2025(BaseData):  # ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
    zbins = {'1': 'pzbin1', '2': 'pzbin2', '3': 'pzbin3', '4': 'pzbin4'}
    samples = {'main': '', 'extended': 'extended_', 'all': ''}
    corrs = {'corrected': 'corr_', 'uncorrected': ''}
    zoutcuts = {'nocut': '', 'cut': 'sigmaz0.05000_'}
    
    path = "/global/homes/c/cpopik/Capybara/Data/Hadzhiyska2024"
    beamfile = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220/ilc_beam.txt"  # TODO 2
    respfile = None  # TODO 2
    dustfile = None  # TODO 3
    freq = 90  # check this?


    def __init__(self, spefs):       
        self.checkspefs(spefs, required=['zbin', 'sample', 'corr', 'zoutcut'])
         
        self.thetas = np.load(f"{self.path}/Fig2_sim.npz")['theta_arcmins']
        self.Tksz_Illustris1 = np.load(f"{self.path}/Fig2_sim.npz")['gas_illustris']
        self.Tksz_TNG300 = np.load(f"{self.path}/Fig2_sim.npz")['dm_tng']
        self.Tksz = np.load(f"{self.path}/Fig2_sim.npz")['signal']
        
        filename = f"{self.path}/Fig1_Fig8_{self.sample}dr10_allfoot_perbin_{self.zoutcut}dr6_{self.corr}{self.zbin}.npz"
        self.kSZdata = np.load(filename)['prof']
        self.kSZcov = np.load(filename)['cov']
        
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, uniteless]
        
        # Placeholders
        self.respfile = f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/responses/act_planck_dr5.01_s08s18_AA_f090_daynight_response_tsz.txt"
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = -self.resp_data 
        self.dustprof = np.interp(self.thetas, roughdustx, roughdusty)  # [muK*arcmin^2]



class Liu2025(BaseData):  # ACT DR6 maps stacked on DESI LRGs for cross-correlation (arxiv.org/abs/2502.08850)
    zbins = {'1':'pz1','2':'pz2', '3':'pz3', '4':'pz4'}
    dBetas = {'fiducial':'fiducial', '1.2':'dBeta_1.2_10.7', '1.4':'dBeta_1.4_10.7','1.6':'dBeta_1.6_10.7'}
    
    freq = 150
    dirname = "/global/homes/c/cpopik/Data/StackedProfiles_outputs_for_Nick"
    beamfile = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220/ilc_beam.txt" 
    zdistfile = "/global/homes/c/cpopik/Capybara/Data/Liu2025/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"
    # TODO 2
    respfile = None  # TODO 2
    dustfile = None  # TODO 3
    

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['zbin', 'dBeta'])
    
        # Find data and covariance files for bin and CIB method
        self.datafname = f"{self.dirname}/DESI_{self.zbin}_act_dr6_{self.dBeta}/diskring_tsz_uniformweight_measured.txt"
        self.covfname = f"{self.dirname}/DESI_{self.zbin}_act_dr6_{self.dBeta}/cov_diskring_tsz_uniformweight_bootstrap.txt"

        # Load in data and convert to arcmin^2
        self.thetas = np.genfromtxt(self.datafname).T[0]  # [arcmin]
        self.tSZdata, self.tSZerr = np.genfromtxt(self.datafname).T[1:3]*u.sr.to(u.arcmin**2)  # [ster]->[arcmin^2]
        self.tSZcov = np.genfromtxt(self.covfname).T*u.sr.to(u.arcmin**2)**2  # [ster]->[arcmin^2]
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, uniteless]
        
        # Placeholders
        self.respfile = f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/responses/act_planck_dr5.01_s08s18_AA_f150_daynight_response_tsz.txt"
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = -self.resp_data 

        self.dustprof = np.interp(self.thetas, roughdustx, roughdusty)  # [muK*arcmin^2]
        
    def tSZdata_in_muK(self, T_CMB, **kwargs):  # Convert data/cov to muK*arcmin^2 to match model, needs a T_CMB value which we won't assume
        self.tSZdata = self.tSZdata*fnu(self.freq, T_CMB)*T_CMB*1e6
        self.tSZerr = -self.tSZerr*fnu(self.freq, T_CMB)*T_CMB*1e6
        self.tSZcov = self.tSZcov*(fnu(self.freq, T_CMB)*T_CMB*1e6)**2
        
    # def windowfunction(self):
        


class Schaan2021:  # ACT DR5 maps stacked on CMASS DR10/DR12 (arxiv.org/abs/2009.05557)
    # Define freq and files for data/cov/beam/response, and dust model
    datafile = "/global/homes/c/cpopik/Capybara/Data/emu4d_match_ACT_profiles.txt"  # TODO 1
    tSZcovfile = "/global/homes/c/cpopik/Capybara/Data/cov_diskring_tsz_varweight_bootstrap.txt"  # TODO 1
    kSZcovfile = "/global/homes/c/cpopik/Capybara/Data/cov_diskring_ksz_varweight_bootstrap.txt"  # TODO 1
    
    beamfiles = lambda self, freq: f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/beams/act_planck_dr5.01_s08s18_f{freq}_daynight_beam.txt"
    respfiles = lambda self, freq: f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/responses/act_planck_dr5.01_s08s18_AA_f{freq}_daynight_response_tsz.txt"
    freq = 150
    limits = {'mhalomax': 1e14}

    def __init__(self, spefs):
        # Load in data, cov, beam, response, and dust profile
        self.thetas = np.genfromtxt(self.datafile).T[0]  # [arcmin]
        self.kSZdata, self.kSZerr = np.genfromtxt(self.datafile).T[1:3]  # kSZdata, kSZerr in ???
        self.tSZdata, self.tSZerr = np.genfromtxt(self.datafile).T[6:8] # [muK*arcmin^2]
        self.tSZcov = np.genfromtxt(self.tSZcovfile).T*u.sr.to(u.arcmin**2)**2  # [muK*ster]->[muK*arcmin^2]
        self.kSZcov = np.genfromtxt(self.kSZcovfile).T*u.sr.to(u.arcmin**2)**2  # [muK*ster]->[muK*arcmin^2]

        self.beamfile = self.beamfiles(self.freq)
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]

        self.respfile = self.respfiles(self.freq)
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = -self.resp_data  # Is negative for some reason
        
        self.dustprof = np.interp(self.thetas, roughdustx, roughdusty)  # [muK*arcmin^2]