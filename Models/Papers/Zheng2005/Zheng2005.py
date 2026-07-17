"""
Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies

ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
arxiv.org/pdf/astro-ph/0408564
"""



import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c

from scipy.special import erf

from Models.Papers.Zehavi2005 import Zehavi2005



    

class Cosmology():
    pass

class HaloModel():
    pass

class HOD(): # Section 3.1. HOD for All Galaxies
    MassDef = "virial"
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(ngbar=self.ngbar,Model=self.Model,numParams=self.numParams).to_dict()
        except: self.p0 = {}
        
        self.Zehavi2005 = Zehavi2005.HOD(inputdict | inputvars)
        
    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        
        if self.numParams=='3P':
            return self.Zehavi2005.Ncen(p)
            
        return (1/2) * (1+erf((self.logM-p['logMmin'])/p['sigmalogM']))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
        
        if self.numParams=='5P':
            return self.Zehavi2005.Nsat(p)
        
        return np.where(10**self.logM>=p['M0'], ((10**self.logM-10**p['M0'])/10**p['logM1prime']), 0)**p['alpha']




# Table 1. HOD Parameters for Galaxy Samples with Different Thresholds of Baryonic Mass
# Note. — Number density and mass are in units of h3Mpc−3 and M⊙, respectively. Columns 3–5 are for the 3-parameter model and Columns 6–10 are for the 5-parameter model (see the text). For the 3-parameter model, Mmin is simply set to be the halo mass at which 〈N 〉M = 0.5, and M1 and α are obtained through a power-law fit to data points with 〈Nsat〉M > 0.1.
class Table1():  
    def __init__(self):
        self.df = pd.read_csv(f"{thispath}/Table1.csv")

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

    
    
#Fig. 1.— Mean occupation number and scatter as a function of halo mass, separated into central and satellite galaxies. Predictions are shown for the  ̄ng = 0.02h3 Mpc−3 samples from the SPH simulation (left panels) and from the SA model (right panels). Lower panels plot the mean occupation numbers of central, satellite, and all galaxies. In the upper panels, circles show 〈N (N − 1)〉1/2/〈N 〉, indicating the width of the probability distribution, for all galaxies (filled circles) and satellite galaxies (open circles). For Poisson P (N |M ), this ratio would be one (dotted line). This figure can be compared to Fig. 4 of K04.
class Fig1(BasePlots2):
    subplots = [[
        dict(name='Fig1c', filename='Fig1c', figsize=(5, 4),
             xlabel=r'$M (M_\odot)$', xlim=(3.15e10, 1e15), xscale='log',
             ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 50), yscale='log'),
        dict(name='Fig1d', filename='Fig1d', figsize=(5, 4),
             xlabel=r'$M (M_\odot)$', xlim=(3.15e10, 1e15), xscale='log',
             ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 50), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Fig. 3.— Parameterized fits to mean occupation functions (top panels) and predicted numbers of galaxy pairs and triplets (middle and bottom panels) for the SPH simulation (left) and the SA model (right). For each model, left panels show results based on 3-parameter fits, which assume sharp cutoff profiles of 〈Ncen〉M and 〈Nsat〉M , and right panels show results of fits with more parameters to model the cutoff profiles (see eqs. [1] and [3]). Fits and predictions are plotted as curves, and circles are measurements from the models.
class Fig3(BasePlots2):
    subplots = [
        [dict(name='Fig3a', filename='Fig3a', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log'),
         dict(name='Fig3b', filename='Fig3b', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log')],
        [dict(name='Fig3c', filename='Fig3c', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log'),
         dict(name='Fig3d', filename='Fig3d', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log')],
    ]

    def __init__(self):
        super().__init__(thispath)


class Studies(BaseStudy):  # Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies, ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    subs = {}
    info = {'MassDef': 'vir',
        }


class HODParamsTable(ParamTable):  # best-fit HOD parameters, Table 1 (same data as Table1.csv above, kept string-typed for exact key matching)
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        self.df = pd.read_csv(filename, dtype={'ngbar': str})


class HODs(BaseHOD, Studies.Zheng2005):  # virial mass
    models = {'model': ['SPH', 'SA'],  # simulation type
              'nparams': ['3P', '5P'],  # number of parameters in model (Zehavi vs Zheng)
              'ng': ['0.02', '0.01', '0.005', '0.0025'],  # number density in h^3 Mpc^-3
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        row = HODParamsTable().getparams(numParams=self.nparams, Model=self.model, ngbar=self.ng).to_dict()
        self.p0 = {'logMmin': row['logMmin'], 'alpha': row['alpha']}
        if self.nparams == '3P':
            self.p0['logM1'] = row['logM1']
        else:
            self.p0['logM1'] = row['logM1prime']
            self.p0['sigmalogM'] = row['sigmalogM']
            self.p0['logM0'] = row['M0']
        
    def Nc(self, logM, logMmin, sigmalogM):  # Eq 1
        return (1/2) * (1+erf((logM-logMmin)/sigmalogM))

    def Ns(self, M, M0, M1, alpha):  # Eq 3
        return np.where(M>=M0, ((M-M0)/M1), 0)**alpha

    def Ncen(self, logM):
        if self.nparams=='3P': func = lambda p: Zehavi2005().Nc(logM, logMmin=p['logMmin'])
        elif self.nparams=='5P': func = lambda p: self.Nc(logM, logMmin=p['logMmin'], sigmalogM=p['sigmalogM'])
        return lambda p={}: func(self.p0 | p)
    
    def Nsat(self, logM):
        if self.nparams=='3P': func = lambda p: Zehavi2005().Ns(10**logM, M1=10**p['logM1'], alpha=p['alpha'])
        elif self.nparams=='5P': func = lambda p: self.Ns(10**logM, M0=10**p['logM0'], M1=10**p['logM1'], alpha=p['alpha'])
        return lambda p={}: func(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    def Fig1(self, width=10, height=4):
        return self.plot(filename=['Fig1c','Fig1d'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
            xlim=(3.15e10, 1e15), ylim=(3.1e-2, 50), xscale='log', yscale='log')
        
    def Fig3(self, width=8, height=6):
        return self.plot(filename=['Fig3a','Fig3b','Fig3c','Fig3d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
            xlim=(1e11, 1e15), ylim=(3.1e-2, 4.9e1), xscale='log', yscale='log')
