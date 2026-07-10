"""
The Impacts of Modeling Choices on the Inference of Circumgalactic Medium Properties from Sunyaev-Zeldovich Observations

arxiv.org/pdf/2103.02469
ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
"""

import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


class Table3_bestfit(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_bestfit.csv"):
        super().__init__(filename)

class Table3_marginalized(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_marginalized.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        


def Fig2(width=14, height=6):
    return BasePlots2(thispath).plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$r/r_{200c}$', r'$r/r_{200c}$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
        xlim=[[8e-2,6.2e0],[8e-2,6.2e0]], ylim=[[5e-31,1.2e-26],[1e-16,1.6e-11]], xscale='log', yscale='log')

def Fig3(width=14, height=6):
    return BasePlots2(thispath).plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$M_h \ [M_\odot]$', r'$\log_10(M^*)\ (M_\odot)$'], ylabel=[r'$M_s \ [M_\odot]$', ''],
        xlim=[[10.8, 16],[10.6,11.8]], ylim=[[7,12.5],[2,5.5e4]], xscale=['linear','linear'], yscale=['linear','log'])

def Fig4row1(width=14, height=6):
    return BasePlots2(thispath).plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
        xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])

def Fig6col1(width=14, height=6):
    return BasePlots2(thispath).plot(filename=['Fig6a','Fig6c'], nrow=1, ncol=2, width=width, height=height,
        xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
        xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])