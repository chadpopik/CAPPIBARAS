"""
Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos

ui.adsabs.harvard.edu/abs/2018AstL...44....8K
arxiv.org/pdf/1401.7329
"""


import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


class Cosmology():
    # Unless otherwise noted, throughout this paper we assume a flat ΛCDM model with parameters Ωm = 1 − ΩΛ = 0.27,Ωb = 0.0469, h = H0/(100 km s−1Mpc−1) = 0.7, σ8 = 0.82 and ns = 0.95 compatible with combined constraints from WMAP, BAO, SNe, and cluster abundance (Vikhlinin et al. 2009b; Komatsu et al. 2011; Hinshaw et al. 2013).
    Om0=0.27
    Ob0=0.0469
    h=0.7
    sigma8=0.82
    ns=0.95
    


class HaloModel():
    # Total masses are defined within radius enclosing a particular overdensity (500 or 200) with respect to the critical density at redshift of observation, which is indicated by a corresponding subscript (M500 or M200).
    pass


class Table3(ParamTable):
    # TABLE 3 Parameters of best fit M∗ − M parametrization at z . 0.1
    def __init__(self, filename=f"{thispath}/table3.csv"):
        super().__init__(filename)

def Fig4(width=4, height=4):
    return BasePlots2(thispath).plot(filename='Fig4', width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
        xlim=(1.8e13, 2e15), ylim=(5e10, 2.5e13), xscale='log', yscale='log')

def Fig7(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig7', width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}(<r_{500}) \ [M_\odot]$',
        xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

def Fig8(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig8', width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{sat}}(<r_{500}) \ [M_\odot]$',
        xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

def Fig9(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig9', width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}}/(M_{*, \text{BCG}}+M_{*, \text{sat}})$',
        xlim=(2e13, 2e15), ylim=(0, 1), xscale='log', yscale='log')

def Fig10(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig10', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}} \ [M_\odot]$',
        xlim=(1e10, 4e15), ylim=(1e8, 2e13), xscale='log', yscale='log')

def Fig11(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig11', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
        xlim=(1e10, 4e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

def Fig12(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig12', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
        xlim=(1e10, 4e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

def Fig13(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig13', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
        xlim=(1e10, 1e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

def Fig14(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig14', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
        xlim=(1e10, 1e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

def Fig15(width=15, height=5):
    return BasePlots2(thispath).plot(filename=['Fig15a','Fig15b','Fig15c'], nrow=1, ncol=3, width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$',
        xlim=(1.3e11, 1.6e15), ylim=(3e-3, 1), xscale='log', yscale='log')

def Fig16(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig16', width=width, height=height,
        xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
        xlim=(1.8e13, 2e15), ylim=(2.5e10, 2.5e13), xscale='log', yscale='log')

def Fig17(width=6, height=6):
    return BasePlots2(thispath).plot(filename='Fig17', width=width, height=height,
        xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_* \ [M_\odot]$',
        xlim=(1e9, 2e15), ylim=(1e7, 5e12), xscale='log', yscale='log')