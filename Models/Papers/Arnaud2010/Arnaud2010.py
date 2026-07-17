"""
The universal galaxy cluster pressure profile from a representative sample of nearby systems (REXCESS) and the YSZ - M500 relation

ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
arxiv.org/pdf/0910.1234
"""

import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


class Cosmology():
    # 1. We adopt a ΛCDM cosmology with H0 = 70 km/s/Mpc,ΩM = 0.3 and ΩΛ = 0.7. 
    H0 = 70
    Om0 = 0.3
    Ol0 = 0.7
    

class HaloModel():
    # 1. Here and in the following, Mδ and Rδ are the total mass and radius corresponding to a density contrast, δ, as compared to ρc(z), the critical density of the universe at the cluster redshift: Mδ = (4π/3)δρc(z)R3δ .M500 corresponds roughly to the virialised portion of clusters, and is traditionally used to define the ’total’ mass.
    MassDef = '500c'
    
    Concentration = 'Constant'
    

# GNFW model of the universal pressure profile (green line). It is derived by fitting the observed average scaled profile in the radial range [0.03–1]R500 , combined with the average simulation profile beyond R500 (red line).
class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8', filename='Fig8', figsize=(6, 5),
             xlabel=r'Radius $(R_{500})$', xlim=(7.5e-3, 5.2), xscale='log',
             ylabel=r'$P/P_{500}$', ylim=(2.5e-4, 3.4e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Studies(BaseStudy):  # The universal galaxy cluster pressure profile from a representative sample of nearby systems (REXCESS) and the YSZ - M500 relation, ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
    subs = {}
    info = {
        # Cosmological Parameters, 1p-1
        'h':0.7, 'Om0':0.3, 'Ol0':0.7, 'Concentration': 'Constant', 'MassDef':'500c',
    }


class ParamsTable(ParamTable):  # Eq 12, best-fit parameter sets
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)


class Profiles(BaseProfile, Studies.Arnaud2010):  # Pressure Profile fit to REXCESS cluseters with XMM-Newton data
    models = {'model': ['norm', 'ST', 'coolcore', 'disturbed'],}  # different best-fit parameter sets
    fixedparams = {
        'alpha_P': 0.12,  # mass dependence
        }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = self.fixedparams | ParamsTable().getparams(model=self.model).to_dict()
        self.h70 = self.h*100/70

    def P500(self, z, logM500c, units='cosmo'):  # Eq 5
        val = 1.65e-3 * (self.H(z)/self.H0)**(8/3) * (10**logM500c/(3e14/self.h70))**(2/3) *self.h70**2 *u.keV/u.cm**3
        return val.to(self.units('pres', units))
    
    def PGNFW(self, x, gamma, alpha, beta, P0, c500):  # Eq 11
        return P0 / ((x*c500)**gamma * (1+(x*c500)**alpha)**((beta-gamma)/alpha))

    def alphaPp(self, x, alpha_P):  # Eq 13
        return 0.10-(alpha_P+0.10)*(x/0.5)**3/(1+x/0.5)**3
    
    def mdep(self, x, logM500c, alphaPprime=False):  # mass dependence factor, Eq 13?
        alphaPp = lambda p: self.alphaPp(x, p) if alphaPprime else 0
        return lambda p: (10**logM500c/(3e14/self.h70))**(p['alpha_P']+alphaPp(p))
    
    def Pressure(self, r, z, logM500c, units='cosmo', alphaPprime=False):  # Eq 4/8/10/13
        r, z, logM500c = self.setdim(r, z, logM500c)  # set proper dimensions
        P500 = self.P500(z, logM500c, units)
        x = r*u.Mpc/(self.r500c(z, logM500c))
        PGNFW = lambda p: self.PGNFW(x, gamma=p['gamma'], alpha=p['alpha'], beta=p['beta'], P0=p['P0']*self.h70**(-3/2), c500=p['c500'])
        mdep = self.mdep(x, logM500c, alphaPprime)
        return lambda p={}: P500*mdep(self.p0 | p)*PGNFW(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
    def Fig8(self, width=6, height=5):
        return self.plot(filename='Fig8', width=width, height=height,
            xlabel=r'Radius $(R_{500})$', ylabel=r'$P/P_{500}$',
            xlim=(7.5e-3, 5.2), ylim=(2.5e-4, 3.4e2), xscale='log', yscale='log')
