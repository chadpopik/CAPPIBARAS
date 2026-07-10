"""
Probing cosmic velocities with the pairwise kinematic Sunyaev-Zel'dovich signal in DESI Bright Galaxy Sample DR1 and ACT DR6

ui.adsabs.harvard.edu/abs/2026PhRvD.113f3565H
arxiv.org/pdf/2510.14135
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

from Models.Papers.Hadzhiyska2025B import Hadzhiyska2025B


    # info = Planck2018.info | { # says it uses Planck2018 cosmology
    #     "MassDef": 'vir',

        
    # }


class Cosmology():
    pass

class HaloModel():
    pass

class HOD(Hadzhiyska2025B.HOD):
    # The halo mass definition adopted in the AbacusSummitsimulations is the ‘virial mass’ one from Ref. [128].
    MassDef = 'vir'
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(subsample=self.subsample).to_dict()
        except: self.p0 = {}
    
    # In brief, we use a vanilla Halo Occupation Distribution (HOD) model applied to the AbacusSummit simulations at fixed redshift z = 0.3 [Maksimova2021], appropriate for BGS. The HOD is described by five free parameters, sampled using a Latin Hypercube with 1000 realizations as in the initial work. We adopt the AbacusHOD prescription within theabacusutils package 1. 
    
    
    


# TABLE I. Summary of the BGS stellar-mass threshold samples, including values of the 5 HOD parameters, inferred satellite fraction, mean halo mass, linear bias, and χ2 null with 4 degrees of freedom. Error bars denote 1σ uncertainties. As estimated in Ref. [123], the systematic and model errors expected from this type of analysis are ∼7%.
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        

    
# FIG. 2. Top panel: Number of BGS galaxies, with a stellar mass threshold of log10(M⋆/M⊙) > 10, in Y1 as a function of stellar mass in differential bins. The distribution peaks near log10(M⋆/M⊙) ≈ 10.75, reflecting the stellar-mass completeness of the sample at the median survey redshift. 
def Fig2a(width=5, height=4):
    return BasePlots2(thispath).plot(filename='Fig2a', width=width, height=height,
        xlabel=r'$\log (M/M_\odot)$', ylabel=r'$N_\text{gal}$',
        xlim=(9.9, 12.15), ylim=(-0.6e3, 12.7e3), xscale='linear', yscale='linear')

# Bottom panel: Number of BGS galaxies per redshift bin for six cumulative stellar-mass thresholds. Higher-mass samples trace progressively higher-redshift populations, with the mean redshift increasing from  ̄z ≈ 0.25 at log10(M⋆/M⊙) > 10 to  ̄z ≈ 0.36 at log10(M⋆/M⊙) > 11.25. These trends highlight the tradeoff between number density and redshift reach when selecting stellar-mass subsamples. The number of galaxies in each mass bin is given in Table II.
def Fig2b(width=5, height=4):
    return BasePlots2(thispath).plot(filename='Fig2b', width=width, height=height,
        xlabel=r'$z$', ylabel=r'$N_\text{gal}$',
        xlim=(0.06, 0.6), ylim=(-0.5e3, 11.25e3), xscale='linear', yscale='linear')