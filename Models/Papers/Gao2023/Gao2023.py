"""
The DESI One-Percent Survey: Constructing Galaxy-Halo Connections for ELGs and LRGs Using Auto and Cross Correlations

ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
arxiv.org/pdf/2306.06317
"""


import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


    # info = {        
    #     # fixed cosmo info
    #     'h':0.71, 'Om0':0.268, 'Ol0':0.732,
    #     'mdef':'vir',  # Current Virial Mass
    #     'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
    # }
    
class Cosmology():
    pass

class HaloModel():
    pass
    
    
def Fig7a(width=8, height=6):
    return BasePlots2(thispath).plot(filename='Fig7a', width=width, height=height,
        xlabel=r'$\log(M_h) \ [M_\odot /h]$', ylabel=r'$\log(M_*) \ [M_\odot]$',
        xlim=(10, 15), ylim=(7.5, 12), xscale='linear', yscale='linear')

def Fig2(width=10, height=3.5):
    return BasePlots2(thispath).plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'$\log M_* \ [M_\odot]$', ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$',
        xlim=[(9, 12.5), (7, 12.5)], ylim=(5e-7, 2e-3), xscale='linear', yscale='log')