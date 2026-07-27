"""
The Average Star Formation Histories of Galaxies in Dark Matter Halos from z = 0-8

ui.adsabs.harvard.edu/abs/2013ApJ...770...57B
arxiv.org/pdf/1207.6105
"""



import sys,os
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Behroozi2013")

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c



# def Behroozi(self, logMh, logM1, logeps, alpha, delta, gamma):
#     Mh, M1, eps = 10**logMh, 10**logM1, 10**logeps
#     f = lambda x : -np.log10(10**(alpha*x)+1) + delta*(np.log10(1+np.exp(x)))**gamma/(1+np.exp(10**(-x)))
#     Ms = 10**( np.log10(eps*M1) + f(np.log10(Mh/M1)) - f(0) )
#     return np.log10(Ms)