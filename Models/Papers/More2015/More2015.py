"""
The Weak Lensing Signal and the Clustering of BOSS Galaxies. II. Astrophysical and Cosmological Constraints

ui.adsabs.harvard.edu/abs/2015ApJ...806....2M
arxiv.org/pdf/1407.1856
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


    # subs = {'mbin': ['MA', 'MB', 'MC']}
    # info = {
    #     # Free cosmological parameters, Table 1
    #     "Om0": {"MA": 0.310, "MB": 0.306, "MC": 0.304},
    #     "sigma8": {"MA": 0.785, "MB": 0.839, "MC": 0.813},
    #     "100*Ob0h2": {"MA": 2.228, "MB": 2.226, "MC": 2.222},
    #     "ns": {"MA": 0.964, "MB": 0.963, "MC": 0.961},
    #     "h": {"MA": 0.703, "MB": 0.700, "MC": 0.695},
    #     # Sample info, Section 2p2
    #     'logMsMin': {"MA": 11.10, "MB": 11.30, "MC": 11.40},
    #     'logMsMax': {"MA": 12.00, "MB": 12.0, "MC": 12.0},
    #     'Ngal': {"MA": 400916, "MB": 196578, "MC": 116682},
    #     'ngal': {"MA": 3e-4, "MB": 1.5e-4, "MC": 0.8e-4},  # (Mpc/h)^{-3}
    #     # Model definitions, Section 3.1pLast
    #     'mdef': '200m',  # M200b, 200 times overdense wrt background matter density
    #     'HMFModel':'Tinker08' ,'BiasModel': 'Tinker10', 'ConcModel':'Maccio08',
        
    #     # Free model parameters, Table 1
    #     "M_stellar_11": {"MA": 0, "MB": 0, "MC": 0},  # describes the average stellar mass of galaxies, [10^11 h^(-2) Msun]
    #     "R_c": {"MA": 0.98, "MB": 1.01, "MC": 1.02},  # normalization of the concentration mass relation with respect to the one obtained from simulations
    #     "psi": {"MA": 0.93, "MB": 0.93, "MC": 0.94},  # nuisance parameters
    # }

    # # info['MhM_stellar_11Min'] = cycle(info['M_stellar_11'], lambda M, h=info['h']: M*u.Msun/h)  # TODO: how to handle multiple h values??
    # info['Ob0h2'] = cycle(info['100*Ob0h2'], lambda o: o/100)
    
    
class Table1():  # HOD Parameters for Galaxy Samples with Different Thresholds of Baryonic Mass
    # Number density and mass are in units of h3Mpc−3 and M⊙, respectively
    def __init__(self):
        dfraw = pd.read_csv(f"{thispath}/Table1.csv")
        
        self.df, self.df_errup, self.df_errdown = splittable(dfraw)

    def getcol(self, key):
        return self.df[key].values

    def getparams(self, **keys):
        df = self.df.copy()
        for k,v in keys.items():
            if k not in df.columns: 
                print(f"key {k} not in {df.columns.to_list()}")
                pass
            elif v in df[k].values: df = df.set_index(k).loc[v]
            else: print(f"Value {v} not in {np.unique(df[k].values)}")
        return df
    
    
def Fig2(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig2', width=width, height=height,
        xlabel=r'$M\ [h^{-1} M_\odot]$', ylabel=r'$\langle N\rangle_M$',
        xlim=(1e12, 3.1e15), ylim=(1e-1, 1e1), xscale='log', yscale='log')

def Fig4(width=12, height=4):
    return BasePlots2(thispath).plot(filename=['Fig4a','Fig4b','Fig4c'], nrow=1, ncol=3, width=width, height=height,
        xlabel=r'$M \ [h^{-1}M_\odot]$', ylabel=r'$\langle N\rangle_M$',
        xlim=(3.2e11, 3.1e15), ylim=(1e-2, 2.8e2), xscale='log', yscale='log')