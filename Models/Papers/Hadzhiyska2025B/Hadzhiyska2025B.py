"""
Missing baryons recovered: A measurement of the gas fraction in galaxies and groups with the kinematic Sunyaev-Zel'dovich effect and CMB lensing

ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
arxiv.org/pdf/2507.14136
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))

    
    
def Cosmology():
    # = III.A As this analysis is performed at fixed cosmology, we employ the fiducial cosmology boxes which have cosmological parameters set to their Planck 2018 values: Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0.
    # info = {'h': 0.6736, 'MassDef': 'vir'
    # }
    # #Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0
    pass

def HaloModel():
    # III.A. All halo masses quoted in this work correspond to the mass definition adopted by the AbacusSummit halo finder CompaSO [46], which defines the virial mass using the spherical collapse model and the fitting formulae from Bryan and Norman [47].
    MassDef = 'vir'

# III.B. To model the distribution of Luminous Red Galaxies (LRGs) within dark matter halos, we adopt a standard (‘vanilla’) fiveparameter Halo Occupation Distribution (HOD) framework [Zheng et al 2005]. n this model, the mean number of central and satellite galaxies in a halo of mass 𝑀 is given by:
class HOD(): 
    MassDef = 'vir'
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1_2_3().getparams(Sample=self.Sample,Bin=self.Bin).to_dict()
        except: self.p0 = {}

    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logM-p['logMcut'])/(2*p['sigmalogM'])))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=p['kappa']*10**p['logMcut'], ((10**self.logM-p['kappa']*10**p['logMcut'])/10**p['logM1']), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)

    
# TABLE I. Best-fit values and 68% confidence intervals for the five HOD parameters and three derived parameters: comoving number density  ̄𝑛 (in [Mpc/ℎ]−3), satellite fraction 𝑓sat, and mean halo mass⟨𝑀halo⟩ (in 𝑀⊙ /ℎ). Results are shown for each of the three tracer samples: Main LRGs, Extended LRGs, and BGS. All mass units are in 𝑀⊙ /ℎ. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 7% for the systematic bias on the mean halo mass, as described in the main text.
# TABLE II. Best-fit HOD and derived parameters for the Main LRG sample, shown across four redshift bins. The redshift bins correspond to: Bin 1: 0.4, 0.54, 0.713, 0.86, 𝑧1 < 𝑧 < 0.54, Bin 2: 0.54 < 𝑧 <0.713, Bin 3: 0.713 < 𝑧 < 0.86, Bin 4: 0.86 < 𝑧 < 1.024. All mass units are in 𝑀⊙ /ℎ, and comoving number densities are in [Mpc/ℎ]−3. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 5% for the systematic bias on the mean halo mass (see Section III D).
# TABLE III. Same as Table II, but for the Extended LRG sample. The redshift bins are identical to those used in Table II.
class Table1_2_3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1_2_3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class Studies(BaseStudy):  # Probing cosmic velocities with the pairwise kinematic Sunyaev-Zel'dovich signal in DESI Bright Galaxy Sample DR1 and ACT DR6, ui.adsabs.harvard.edu/abs/2025arXiv251014135H
    subs = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
    }

    info = Planck2018.info | { # says it uses Planck2018 cosmology
        "MassDef": 'vir',
        "logMsMin": {"M1": 11.861, "M2": 11.918, "M3": 11.831, "M4": 11.750, "M5": 11.947, "M6": 12.183},
        "fsat": {"M1": 0.123, "M2": 0.147, "M3": 0.206, "M4": 0.295, "M5": 0.333, "M6": 0.343},
        "logMhMean": {"M1": 13.135, "M2": 13.184, "M3": 13.266, "M4": 13.310, "M5": 13.370, "M6": 13.456},
        "blin": {"M1": 1.155, "M2": 1.190,  "M3": 1.258, "M4": 1.314, "M5": 1.384, "M6": 1.475},
        
    }


class ParamsTable(ParamTable):  # characteristic HOD parameters, per mass bin
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)


class HODs(BaseHOD, Studies.Hadzhiyska2025B):  #
    models = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = ParamsTable().getparams(mbin=self.mbin).to_dict()

    def Ncen(self, logM):  # Eq 36
        func = lambda p: Zheng2005().Nc(logM-np.log10(self.h), logMmin=p['logMcut'], sigmalogM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005().Ns(10**logM/self.h, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)
