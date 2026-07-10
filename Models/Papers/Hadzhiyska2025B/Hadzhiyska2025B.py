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
from Models.Plots import BasePlots2, splittable, ParamTable
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