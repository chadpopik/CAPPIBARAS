"""
Atacama Cosmology Telescope: Modeling the gas thermodynamics in BOSS CMASS galaxies from kinematic and thermal Sunyaev-Zel'dovich measurements

ui.adsabs.harvard.edu/abs/2021PhRvD.103f3514A
arxiv.org/pdf/2009.05558
"""

import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c



class Cosmology():
    pass
    # info = {
    #     # cosmological parameters, Ip10, IIA.Ap4/p5
    #     'Om0': 0.25, 'Ob0': 0.044, 'OL0': 0.75, 'h': 0.7,  
    #     'v_rms': 1.06e-3, 'XH':0.76,   # RMS of peculiar velocites [v/c] and hydrogen mass fraction
    #     'T_CMB': 2.725*u.K,  # mop-c-gt, mopc.py
    #     # Model implementation, Ip10, Appendix A p2
    #     'MassDef': 'fof', 'MassFunc':'sheth99', 'HaloBias':'sheth01',
    # }

class HaloModel():
    pass


def Fig6_row1(width=16, height=6):
    return BasePlots2(thispath).plot(filename=['Fig6a','Fig6b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R [\text{Mpc}]$', r'$R [\text{Mpc}]$'],
        ylabel=[r'$\rho_\text{gas} \ [\text{g cm}^{-3}]$', r'$P_\text{th} \ [\text{erg cm}^{-3}]$'],
        xlim=[[7.7e-2, 1.25e1],[7.7e-2, 1.25e1]], ylim=[[5e-32,4.1e-26],[1.5e-16,2.5e-12]],
        xscale='log', yscale='log')

def Fig6_row2(width=16, height=6):
    return BasePlots2(thispath).plot(filename=['Fig6c','Fig6d'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
        ylabel=[r'$\rho^\text{2D}_\text{gas} \ [\text{g cm}^{-3}] \cdot \text{Mpc}$', r'$P^\text{2D}_\text{th} \ [\text{erg cm}^{-3}] \cdot \text{Mpc} $'],
        xlim=[[0.8,6.1],[0.8,6.1]], ylim=[[6.9e-5,1.5e-3],[3e10,9.5e11]],
        xscale='linear', yscale='log')

def Fig6_row3(width=15, height=5):
    return BasePlots2(thispath).plot(filename=['Fig6e','Fig6f'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
        ylabel=[r'$T_\text{kSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$'],
        xlim=[[0.75,6.25],[0.75,6.25]], ylim=[[1.2e-1,4e1],[-22,1.25]],
        xscale=['linear','linear'], yscale=['log','linear'])

def Fig7(width=16, height=6):
    return BasePlots2(thispath).plot(filename=['Fig7a','Fig7b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$r/R_\text{200c}$', r'$r/R_\text{200c}$'],
        ylabel=[r'$\rho_\text{gas}(r) / \rho_c$', r'$P_\text{th} (r) / / P_{200}$'],
        xlim=[[7.3e-3,3.2e1],[7.3e-3,3.2e1]], ylim=[[2.7e-4,8.2e3],[5.5e-8,8.2e1]],
        xscale='log', yscale='log')

def Fig11_row2(width=20, height=5):
    return BasePlots2(thispath).plot(filename=['Fig11b','Fig11c','Fig11d'], nrow=1, ncol=3, width=width, height=height,
        xlabel=[r'$R [\text{arcmin}]$',r'$R [\text{arcmin}]$',r'$R [\text{arcmin}]$'],
        ylabel=[r'$I \ [\text{kJy/sr}]$',r'$I \ [\text{kJy/sr}]$',r'$I \ [\text{kJy/sr}]$'],
        xlim=[[1.5,6.2],[1.5,6.2],[1.5,6.2]], ylim=[[-0.2,1.6],[-0.3,2.8],[-0.3,2.7]],
        xscale='linear', yscale='linear')

def Fig11_row3(width=15, height=5):
    return BasePlots2(thispath).plot(filename=['Fig11e','Fig11f'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
        ylabel=[r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$'],
        xlim=[[1.5,6.2],[1.5,6.2]], ylim=[[-22.5,3.5],[-27.5,2.5]],
        xscale='linear', yscale='linear')