"""
Constraining the galaxy-halo connection of infrared-selected u n W I S E galaxies with galaxy clustering and galaxy-CMB lensing power spectra

ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
arxiv.org/pdf/2203.12583
"""


import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, splittable
thispath = os.path.dirname(os.path.abspath(__file__))




class Cosmology():
    pass
    # 1. Throughout this analysis, we assume a flat ΛCDM cosmology with Planck 2018 best-fit parameter values (last column of Table II of Ref. [16]): ωcdm = 0.11933, ωb = 0.02242, H0 = 67.66 km/s/Mpc, ln(1010As) = 3.047 andns = 0.9665 with kpivot = 0.05 Mpc−1, and τreio = 0.0561.
    
    # 1. In our analysis, we work in units of M /h for masses and we adopt the M200chalo mass definition everywhere, i.e., the mass enclosed within the spherical region whose density is 200 times the critical density of the universe, and the corresponding mass-dependent radius r200c, which encloses mass M200c
    
    

    # subs = {'sample':['Blue', 'Green', 'Red'],}

    # info = {
    #     # best fit HOD params
    #     "ASNe7": {"Blue": -0.16, "Green": 1.35, "Red": 27.95},
    #     # Table II
    #     'zMean': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5},
    #     'ndens': {'Blue': 3409, 'Green': 1846, 'Red': 144},
    #     'area': 0.586*4 * np.pi * (180/np.pi)**2*u.deg**2,
        
    #     # fixed cosmo params
    #     'Oc0h2': 0.11933, 'Ob0h2': 0.02242, 'h':0.6766, 'ns':0.9665, 'lnAsn10': 3.047, 'kpivot':0.05, 'tau_reio':0.0561,  # Ip7
    #     # HaloModel choices, Eq 10&30, Section IpLast
    #     'MassDef': '200c', 'Concentration': 'Bhattacharya13', 'MassFunc': 'Tinker08', 'HaloBias': 'Tinker10',
    #     # Other info
    #     'MhMin': 7e8, 'MhMax': 3.5e15,  # Msun/h
    #     'zMin_hmod': 0.005, 'zMax_hmod': 4,
    #     'zMin': 0, 'zMax': 2,
    #     'logM0': 0,
    #         }

    # info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    # info['MhMax'] = cycle(info['MhMax'], lambda M, h=info['h']: M*u.Msun/h)
    # info['ndens'] = cycle(info['ndens'], lambda n: n/u.deg**2)
    
class HaloModel():
    pass
    # B1. dn/(dM ) is the differential number of halos per unit mass and volume, defined by the halo mass function (HMF), where in our analysis we use the Tinker et al. analytical fitting fuction [39]
    
    # In class_sz, we set the mass bounds of the integral to Mmin = 7 × 108 M /hand Mmax = 3.5 × 1015 M /h and the redshift bounds to zmin = 0.005 and zmax = 4, the latter dictated by the upper redshift limit of the unWISE galaxy samples that we analyze.
    
    

def Fig2(width=6, height=4):
    return BasePlots2(thispath).plot(filename='Fig2', width=width, height=height,
        xlabel=r'$z$', ylabel=r'$\frac{1}{N_g^\text{tot}} \frac{dN_g}{dz}$',
        xlim=(0, 4), ylim=(-0.06, 1.3), xscale='linear', yscale='linear')

def Fig8_col1(width=6, height=10):
    return BasePlots2(thispath).plot(filename=['Fig8a','Fig8c','Fig8e'], nrow=3, ncol=1, width=width, height=height,
        xlabel=r'$\ell$', ylabel=r'$10^5 \times C^{gg}_\ell$', xlim=(1e2, 1e3), ylim=(-0.01, 0.75), yscale='linear')

def Fig9(width=15, height=12):
    return BasePlots2(thispath).plot(filename=['Fig9a','Fig9b','Fig9c','Fig9d'], nrow=2, ncol=2, width=width, height=height,
        xlabel=r'Mass [$M_\odot/h$]', ylabel=r'mean number of galaxies',
        xlim=(2e11,5e15), ylim=(1e-2, 1e2), xscale='log', yscale='log')

def Fig15(width=12, height=4):
    return BasePlots2(thispath).plot(filename=['Fig15a','Fig15b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'$\ell$', ylabel=r'$u_\ell^\text{m}$', xlim=(7e1, 1.5e5), ylim=(-0.05, 1.05), xscale='log')