"""
Photometric Objects Around Cosmic Webs (PAC) Delineated in a Spectroscopic Survey. IV. High-precision Constraints on the Evolution of the Stellar-Halo Mass Relation at Redshift z < 0.7

ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
arxiv.org/pdf/2211.02665
"""

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as c

import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


    # subs = {'sample': ['Main', 'LOWZ', 'CMASS'],
    #         'form': ['BP13', 'DP']}


# Section 3.1. Fixed cosmological parameters.
class Cosmology():
    Om0 = 0.268
    Ol0 = 0.732
    sigma8 = 0.831
    h = 0.71


class HaloModel():
    # Eq 8. Virial mass of the halo at the time when the galaxy was last the central dominant object.
    MassDef = 'vir'


class SHMR():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        try: self.p0 = Table2().getparams(model=self.model, redshift=self.redshift).to_dict()
        except: self.p0 = {}

    # Section 3.2. Five-parameter form from Behroozi et al. (2013), used for the BP13 model.
    def log10Mstar_BP13(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        M0, eps = 10**p['log10M0'], 10**p['log10k_log10eps']

        f = lambda x: -np.log10(10**(-p['beta']*x)+1) + p['delta']*(np.log10(1+np.exp(x)))**p['alpha']/(1+np.exp(10**(-x)))
        return np.log10(eps*M0) + f(logMh-p['log10M0']) - f(0)

    # Section 3.2. Double power-law form, used for the DP model.
    def log10Mstar_DP(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        Mh, M0, k = 10**logMh, 10**p['log10M0'], 10**p['log10k_log10eps']

        Mstar = 2*k / ((Mh/M0)**(-p['beta']) + (Mh/M0)**(-p['alpha']))
        return np.log10(Mstar)


# TABLE 2. Best-fit stellar-halo mass relation parameters (BP13 and DP models) in three redshift bins.
class Table2(ParamTable):
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


# Figure 7. The mean stellar-halo mass relations (lines) and 1σ errors (shadows) at different redshift ranges and from both the BP13 (solid lines) and DP models (dashed lines). The horizontal lines indicate the stellar mass limit covered by the observation data at each redshift matched by color.
class Fig7(BasePlots2):
    subplots = [[
        dict(name='Fig7', filename='Fig7', figsize=(6, 6),
             xlabel=r'$M_h \ [h^{-1} \ M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 9. Comparison of the our mean stellar-halo mass function to previous studies at zs ∼ 0.1. Results compared include those from empirical modeling (Behroozi et al. 2013; Moster et al. 2013), from abundance matching (Guo et al. 2010) and from Conditional Stellar Mass Function (CSMF) modeling (Yang et al. 2012).
class Fig9(BasePlots2):
    subplots = [[
        dict(name='Fig9', filename='Fig9', figsize=(8, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* / M_h$', ylim=(1e-4, 1e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig13(BasePlots2):
    subplots = [[
        dict(name='Fig13a', filename='Fig13a', figsize=(6, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
        dict(name='Fig13b', filename='Fig13b', figsize=(6, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
    subs = {'sample': ['Main', 'LOWZ', 'CMASS'],
            'form': ['BP13', 'DP']}
    info = {
        # Fixed cosmo parameters, Section 3.1
        'Om0':0.268, 'Ol0':0.732, 'sigma8':0.831, 'h':0.71,
        # Mass definition, Eq 8 
        'mdef': 'vir',  # virial mass of the halo at the time when the galaxy was last the central dominant object
    }


class ParamsTable(ParamTable):  # best-fit SHMR parameters
    def __init__(self, filename=f"{thispath}/params.csv"):
        super().__init__(filename)


class SHMRs(BaseSHMR, Studies.Xu2023):  # SDSS DR7 Main and SDSSIII BOSS DR12 LOWZ & CMASS
    models = {'sample': ['Main', 'LOWZ', 'CMASS'],  # galaxy sample
            'form': ['BP13', 'DP'],  # form of SHMR
            }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        row = ParamsTable().getparams(form=self.form, sample=self.sample).to_dict()
        self.p0 = {k: v for k, v in row.items() if pd.notna(v)}

    def SHMR(self, logMh):
        self.require(['form', 'sample'])
        if self.form=='BP13': return self.SHMR_BP13(logMh)
        elif self.form=='DP': return self.SHMR_DP(logMh)

    def SHMR_BP13(self, logMh):
        func = lambda p: self.Behroozi(logMh-np.log10(self.h), logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        return lambda p={}: func(self.p0 | p)

    def SHMR_DP(self, logMh):
        func = lambda p: self.DoublePowerLaw(logMh-np.log10(self.h), logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
    def Fig9(self, width=8, height=6):
        return self.plot(filename='Fig9', width=width, height=height,
            xlabel=r'$M_h \ [M_\odot]$', ylabel=r'$M_* / M_h$',
            xlim=(1e10, 1e15), ylim=(1e-4, 1e-1), xscale='log', yscale='log')

    def Fig7(self, width=6, height=6):
        return self.plot(filename='Fig7', width=width, height=height,
            xlabel=r'$M_h \ [h^{-1} \ M_\odot]$', ylabel=r'$M_* \ [M_\odot]$',
            xlim=(1e10, 1e15), ylim=(1e6, 1e12), xscale='log', yscale='log')
