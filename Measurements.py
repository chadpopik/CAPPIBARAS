"""
Collection of measurements/covariances from different stacking/cross-correlation studies and corresponding components needed for forward modelling and inference, such as the beam, response, frequency, and dust model. 
Classes should have consistently titled attributes and units.

TODO 1: Find better source of Schaan measurements/covariances and dust model (just using Emily's TNG for now). Also why is the covariances off from the error, shouldn't they be the same?
TODO 2: What are the units on the beam response and how should it be converted? Using it raw isn't right.
TODO 3: Find Liu response, and probably have to figure out conversions same as TODO 2.
TODO 4: Find dust model to use for Liu, just using Emily's TNG file for now.
TODO 5: Ask about dust file, should it be negative?
TODO 6: Add boryana measurements https://zenodo.org/records/12633573
"""

from Basics import *
from ForwardModel import fnu


class Liu2025:  # ACT DR6 maps stacked on DESI LRGs for cross-correlation (arxiv.org/abs/2502.08850)
    # Define options & freq for measurements, directory location for data, files for beam/response, and dust model
    bins = ['pz1', 'pz2', 'pz3', 'pz4']
    dBetas = ['fiducial', 'dBeta_1.2_10.7', 'dBeta_1.4_10.7', 'dBeta_1.6_10.7']
    freq = 150
    dirname = "/global/homes/c/cpopik/Git/Capybara/Data/StackedProfiles_outputs_for_Nick"
    beamfile = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220/ilc_beam.txt"
    respfile = None  # TODO 3
    dustfile = "/global/homes/c/cpopik/Git/Capybara/Data/fig6_TNG_H_dust.txt"  # TODO 4

    def __init__(self, bin, dBeta):
        # Find data and covariance files for bin and CIB method
        self.datafname = f"{self.dirname}/DESI_{bin}_act_dr6_{dBeta}/diskring_tsz_uniformweight_measured.txt"
        self.covfname = f"{self.dirname}/DESI_{bin}_act_dr6_{dBeta}/cov_diskring_tsz_uniformweight_bootstrap.txt"

        # Load in data and convert to arcmin^2
        self.thetas = np.genfromtxt(self.datafname).T[0]  # [arcmin]
        self.tSZdata, self.tSZerr = np.genfromtxt(self.datafname).T[1:3]*u.sr.to(u.arcmin**2)  # [ster]->[arcmin^2]
        self.tSZcov = np.genfromtxt(self.covfname).T*u.sr.to(u.arcmin**2)**2  # [ster]->[arcmin^2]
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, uniteless]
        self.resp_ells, self.resp_data = self.beam_ells, np.ones(self.beam_ells.shape)  # [ells, unitless]
        self.dustprof = -np.genfromtxt(self.dustfile).T[1]  # [muK*arcmin^2], TODO 5
        
    def tSZdata_in_muK(self, T_CMB):  # Convert data/cov to muK*arcmin^2 to match model, needs a T_CMB value
        self.tSZdata = self.tSZdata*fnu(self.freq, T_CMB)*T_CMB*1e6
        self.tSZerr = self.tSZerr*fnu(self.freq, T_CMB)*T_CMB*1e6
        self.tSZcov = self.tSZcov*(fnu(self.freq, T_CMB)*T_CMB*1e6)**2


class Schaan2021:  # ACT DR5 maps stacked on CMASS DR10/DR12 (arxiv.org/abs/2009.05557)
    # Define freq and files for data/cov/beam/response, and dust model
    datafile = "/global/homes/c/cpopik/Git/Capybara/Data/emu4d_match_ACT_profiles.txt"  # TODO 1
    tSZcovfile = "/global/homes/c/cpopik/Git/Capybara/Data/cov_diskring_tsz_varweight_bootstrap.txt"  # TODO 1
    kSZcovfile = "/global/homes/c/cpopik/Git/Capybara/Data/cov_diskring_ksz_varweight_bootstrap.txt"  # TODO 1
    dustfile = "/global/homes/c/cpopik/Git/Capybara/Data/fig6_TNG_H_dust.txt"  # TODO 1
    freq = 150

    def __init__(self):
        # Load in data, cov, beam, response, and dust profile
        self.thetas = np.genfromtxt(self.datafile).T[0]  # [arcmin]
        self.kSZdata, self.kSZerr = np.genfromtxt(self.datafile).T[1:3]  # kSZdata, kSZerr in ???
        self.tSZdata, self.tSZerr = np.genfromtxt(self.datafile).T[6:8] # [muK*arcmin^2]
        self.tSZcov = np.genfromtxt(self.tSZcovfile).T*u.sr.to(u.arcmin**2)**2  # [muK*ster]->[muK*arcmin^2]
        self.kSZcov = np.genfromtxt(self.kSZcovfile).T*u.sr.to(u.arcmin**2)**2  # [muK*ster]->[muK*arcmin^2]

        self.beamfile = f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/beams/act_planck_dr5.01_s08s18_f{self.freq}_daynight_beam.txt"
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]

        self.respfile = f"/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary/responses/act_planck_dr5.01_s08s18_AA_f{self.freq}_daynight_response_tsz.txt"
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = self.resp_data/self.resp_data[0]  # TODO 2
        
        self.dustprof = -np.genfromtxt(self.dustfile).T[1]  # [muK*arcmin^2]



Classes = {
    "Liu2025": Liu2025,
    "Schaan2021": Schaan2021,
}

def get_Class(class_name):
    print("Loading Data")
    try:
        return Classes[class_name]
    except KeyError:
        raise ValueError(f"Unknown class: {class_name}. Choose from {list(Classes.keys())}")