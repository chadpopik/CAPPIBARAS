"""
The DESI one-per cent survey: exploring the halo occupation distribution of luminous red galaxies and quasi-stellar objects with ABACUSSUMMIT

ui.adsabs.harvard.edu/abs/2024MNRAS.530..947Y
arxiv.org/pdf/2306.06314
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c
from scipy.special import erf

import sys,os
from Models.Plots import BasePlots2, splittable, ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))


        

    #     'mdef': '200c',  # M not clear, maybe same as zheng 2005/2007? or cmass?
    #     'MhMin': 1.3e11,  # Msun/h
    #     'zMin': {'LRG1': 0.6, 'LRG2': 0.8, 'QSO': 2.1, 'LRG3': 0.95, 'LRG4': 0.8},
    #     'zMax': {'LRG1': 0.4, 'LRG2': 0.6, 'QSO': 0.8, 'LRG3': 0.8, 'LRG4': 0.95},
    
    
class Cosmology():
    # 1. Throughout this paper, we adopt the Planck 2018 ΛCDM cosmology, specifically the mean estimates of the Planck TT,TE,EE+lowE+lensing likelihood chains: Ω𝑐 ℎ2 = 0.1200,Ω𝑏 ℎ2 = 0.02237, 𝜎8 = 0.811355, 𝑛𝑠 = 0.9649, ℎ = 0.6736,𝑤0 = −1 and 𝑤𝑎 = 0
    Oc0h2 = 0.1200
    Ob0h2 = 0.02237
    sigma8 = 0.811355
    ns = 0.9649
    h = 0.6736
    w0 = -1
    wa = 0

class HaloModel():
    pass


# 4.1 For a LRG sample, the HOD is well approximated by a vanilla model given by (originally shown in Zheng et al. 2007 and referred to as Zheng07 or vanilla later in the text):
class HOD():
    MassDef = '??????'
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table3_4().getparams(Tracer=self.Tracer,Model=self.Model).to_dict()
        except: self.p0 = {}
        
        if "LRG" in self.Tracer: self.Nsat = self.Nsat_LRG
        elif "QSO" in self.Tracer: self.Nsat = self.Nsat_QSO
        
    def Ncen(self, pdict={}, **kwargs): # Eq 4
        p = self.p0 | pdict | kwargs
        return (p['fic']/2) * (1+erf((self.logM-p['logMcut'])/(np.sqrt(2)*p['sigma'])))

    def Nsat_LRG(self, pdict={}, **kwargs):  # Eq 5
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=p['kappa']*10**p['logMcut'], ((10**self.logM-p['kappa']*10**p['logMcut'])/10**p['logM1']), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)
    
    def Nsat_QSO(self, pdict={}, **kwargs):  # Eq 5
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=p['kappa']*10**p['logMcut'], ((10**self.logM-p['kappa']*10**p['logMcut'])/10**p['logM1']), 0)**p['alpha']





class Table3_4(ParamTable):
    # Table 3. LRG and QSO marginalized posteriors, with different models and different measurements. The error bars are 1𝜎 uncertainties. We also display several derived parameters, specifically the marginalized satellite fraction 𝑓sat, the sample completeness 𝑓ic, the average halo mass per galaxy log 𝑀h, and the linear bias𝑏lin. Units of mass are given in ℎ−1 𝑀⊙ .
    # Table 4. The results for the fits to high-z LRG sample with two redshift bin:0.8 < 𝑧 < 0.95 and 0.95 < 𝑧 < 1.1. We show the mean±1𝜎 error for HOD and derived parameters. We also list the average comoving number density in units of 10−4 (ℎ−1Mpc) −3. Masses are in units of ℎ−1 𝑀⊙ .
    def __init__(self, filename=f"{thispath}/Table3_4.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)

    
    
def Fig1(width=6, height=3.5):
    # Figure 1. The DESI One-Percent Survey LRG and QSO mean number density as a function of redshift. The dashed vertical lines show the fiducial LRG redshift bin edges of 𝑧 = 0.6, 𝑧 = 0.8, and the maximum redshift we consider for the QSO sample 𝑧 = 2.1.
    return BasePlots2(thispath).plot(filename='Fig1', width=width, height=height,
        xlabel=r'$z$', ylabel=r'$n(z) \ [h^3 \text{Mpc}^{-3}]$',
        xlim=(0.4, 2.3), ylim=(3e-6, 2e-3), xscale='linear',yscale='log')

def Fig7(width=14, height=6):
    # Figure 7. The LRG HOD best-fit the posterior. The shaded regions correspond to 1 and 2𝜎 posteriors (68% and 95% intervals centered around the median prediction). The horizontal dotted line denotes 𝑁gal = 1.
    return BasePlots2(thispath).plot(filename=['Fig7a','Fig7b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'$M_h [h^{-1} M_\odot]$', ylabel=r'$N_\text{gal}$',
        xlim=(1e12, 1e15), ylim=(1e-2, 1e1), xscale='log', yscale='log')

def Fig10(width=6, height=5):
    # Figure 10. The HOD posterior band (central+satellite) of LRG sample at𝑧 > 0.8. The results from 0.8 < 𝑧 < 0.95 and 0.95 < 𝑧 < 1.1 are shown in red and blue respectively. The shaded regions correspond to 1 and 2 𝜎posteriors.
    return BasePlots2(thispath).plot(filename='Fig10', width=width, height=height,
        xlabel=r'$M_h \ [h^{-1} M_\odot]$', ylabel=r'$N^{(c+s)}_\text{gal}$',
        xlim=(1e12, 1e15), ylim=(3e-2, 1e1), xscale='log', yscale='log')

def Fig14(width=6, height=5):
    # Figure 14. The HOD posterior for the QSO sample. The shaded regions correspond to 1 and 2𝜎 posteriors.
    return BasePlots2(thispath).plot(filename='Fig14', width=width, height=height,
        xlabel=r'$M_h \ [h^{-1} M_\odot]$', ylabel=r'$N_\text{gal}$',
        xlim=(1e12, 1e15), ylim=(3e-2, 1e1), xscale='log', yscale='log')