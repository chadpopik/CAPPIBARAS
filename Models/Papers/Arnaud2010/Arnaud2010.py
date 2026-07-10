"""
The universal galaxy cluster pressure profile from a representative sample of nearby systems (REXCESS) and the YSZ - M500 relation

ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
arxiv.org/pdf/0910.1234
"""

import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
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
    

def Fig8(width=6, height=5):
    # GNFW model of the universal pressure profile (green line). It is derived by fitting the observed average scaled profile in the radial range [0.03–1]R500 , combined with the average simulation profile beyond R500 (red line).
    return BasePlots2(thispath).plot(filename='Fig8', width=width, height=height,
        xlabel=r'Radius $(R_{500})$', ylabel=r'$P/P_{500}$',
        xlim=(7.5e-3, 5.2), ylim=(2.5e-4, 3.4e2), xscale='log', yscale='log')