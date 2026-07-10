"""
Cosmic census: Relative distributions of dark matter, galaxies, and diffuse gas

ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
arxiv.org/pdf/2211.07502
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, splittable
thispath = os.path.dirname(os.path.abspath(__file__))



    #     # fixed cosmological parameters
    #     'h':0.6766, 'Ob0h2':0.02242, 'Oc0h2':0.11933, 'tau':0.0561, 'ns':0.9665, 'sigma8':0.8102, # 5.1p3
    #     'mdef': '200m',  # region in which the average density is ∆ = 200 times the cosmic mean density
    #     'MassFunc': 'Tinker08', 'Concentration':'Dolag04',
    #     'zlims': [0.47, 0.59],  # redshift range of selected galaxies
    #     'zmed': 0.53,  # median redshift
    #     'logMsMin': {'M1': 10.8, 'M2': 11.1, 'M3': 11.25, 'M4': 11.4},  # minimum stellar mass of selected
    #     'c0': 9.59, 'alpha_c': -0.102,  # concentration parameters, Eq47
    # }
    
    # def conc(self, z, logM):  # Eq 47
    #     return self.c0/(1+z) * (10**logM/(10**14))**self.alpha_c

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
    
    
def Fig4(width=9, height=6):
    return BasePlots2(thispath).plot(filename='Fig4', width=width, height=height,
        xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{gg}/(2\pi)$',
        xlim=(3.9e1, 4.1e3), ylim=(2e-2, 1.1e1), xscale='log', yscale='log')

def Fig6(width=12, height=8):
    return BasePlots2(thispath).plot(filename=['Fig6a','Fig6b','Fig6c','Fig6d'], nrow=2, ncol=2, width=width, height=height,
        xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$',
        xlim=(4.1e1, 1.3e3), ylim=(2.5e-9, 2.5e-7), xscale='log', yscale='log')

def Fig8a(width=9, height=6):
    return BasePlots2(thispath).plot(filename='Fig8a', width=width, height=height,
        xlabel=r'$M \ [M_\odot]$', ylabel=r'$\langle N | M \rangle$',
        xlim=(1e12, 2e15), ylim=(5e-4, 3e1), xscale='log', yscale='log')