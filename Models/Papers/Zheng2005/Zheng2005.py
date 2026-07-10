"""
Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies

ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
arxiv.org/pdf/astro-ph/0408564
"""



import sys,os
from Models.Plots import BasePlots2
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
        
        # TODO: work on this
        if self.numParams=='3P':
            self.Zehavi2005 = Zehavi2005.HOD(inputdict | inputvars)
            self.Ncen = lambda pdict={}, **kwargs: self.Zehavi2005.Ncen(self.p0 | pdict | kwargs)
            self.Nsat = lambda pdict={}, **kwargs: self.Zehavi2005.Nsat(self.p0 | pdict | kwargs)

    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logM-p['logMmin'])/p['sigmalogM']))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
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
def Fig1(width=10, height=4):
    return BasePlots2(thispath).plot(filename=['Fig1c','Fig1d'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
        xlim=(3.15e10, 1e15), ylim=(3.1e-2, 50), xscale='log', yscale='log')
        
        
# Fig. 3.— Parameterized fits to mean occupation functions (top panels) and predicted numbers of galaxy pairs and triplets (middle and bottom panels) for the SPH simulation (left) and the SA model (right). For each model, left panels show results based on 3-parameter fits, which assume sharp cutoff profiles of 〈Ncen〉M and 〈Nsat〉M , and right panels show results of fits with more parameters to model the cutoff profiles (see eqs. [1] and [3]). Fits and predictions are plotted as curves, and circles are measurements from the models.
def Fig3(width=8, height=6): 
    return BasePlots2(thispath).plot(filename=['Fig3a','Fig3b','Fig3c','Fig3d'], nrow=2, ncol=2, width=width, height=height,
        xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
        xlim=(1e11, 1e15), ylim=(3.1e-2, 4.9e1), xscale='log', yscale='log')