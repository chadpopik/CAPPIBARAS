"""
DESI luminous red galaxy samples for cross-correlations

ui.adsabs.harvard.edu/abs/2023JCAP...11..097Z
arxiv.org/pdf/2309.06443
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

    # subs = {
    #     'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
    #     'sample' : ['main', 'ext'],  # sample of LRGs
    #     'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    # }
    # info = {
    #     'area': {  # area of survey [deg^2]], 2.1p2/3.2p3
    #             'main':{'combined':16700, 'north':4200, 'south':12500},
    #             'extended':{'combined':230, 'north':100, 'south':130}},
    #     'logMhMean': {'z1': 13.40, 'z2': 13.40, 'z3': 13.24, 'z4': 13.24},  # mean halo mass taken from Yuan 2023 [Msun?], 6.p2
    #     # mean number density, mean redshift, min/max photometrix redshift bounds,  T1/T2/T3
    #     'nGal': {  # [deg^-2]
    #         'main': {'all':600, 'z1': 81.9, 'z2': 148.1, 'z3': 162.4, 'z4': 148.3},
    #         'ext': {'all':1669, 'z1': 185.5, 'z2': 311.0, 'z3': 422.6, 'z4': 438.4},},
    #     'zMean': {
    #         'main': {'z1': 0.470, 'z2': 0.628, 'z3': 0.791, 'z4': 0.924},
    #         'ext': {'z1': 0.467, 'z2': 0.633, 'z3': 0.794, 'z4': 0.929},},
    #     'zpMin': {
    #         'main': {
    #             'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, 'z3': 0.719, 'z4': 0.851},
    #             'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, 'z3': 0.713, 'z4': 0.860},},
    #         'ext': {
    #             'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, '3': 0.719, 'z4': 0.854},
    #             'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, '3': 0.713, 'z4': 0.860},},},
    #     'zpMax': {
    #         'main': {
    #             'north': {'all':1.024, 'z1': 0.545, 'z2': 0.719, 'z3': 0.851, 'z4': 1.024},
    #             'south': {'all':1.020, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.020},},
    #         'ext': {
    #             'north': {'all':1.010, 'z1': 0.545, 'z2': 0.719, 'z3': 0.854, 'z4': 1.010},
    #             'south': {'all':1.000, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.000},},},
    # }
    # for val in ['zpMin', 'zpMax']:  # Assumeding combined uses south limits
    #     for samp in info[val].keys(): 
    #         info[val][samp]['combined'] = info[val][samp]['south']
    # info['area'] = cycle(info['area'], lambda a: a*u.deg**2)
    # info['nGal'] = cycle(info['nGal'], lambda n: n /u.deg**2)
    
    
def Fig2(width=16, height=6):
    return BasePlots2(thispath).plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'Redshift', ylabel=r'$N (\text{deg}^{-2})$',
        xlim=(0.15, 1.25), ylim=[(0, 28), (0, 75)], xscale='linear', yscale='linear')
    
def Fig3(width=16, height=6):
    return BasePlots2(thispath).plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'Redshift', ylabel=r'$10^{3} n(z) (h^{3} \ \text{Mpc}^{-3})$',
        xlim=(0.15, 1.1), ylim=[(0, 0.65), (0, 1.65)], xscale='linear', yscale='linear')