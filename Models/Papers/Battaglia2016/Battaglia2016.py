"""
The tau of galaxy clusters

ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
arxiv.org/pdf/1607.02442
"""


import sys,os
from Models.Plots import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c



class Cosmology():
    
    
    #     info = {
    #     # sim's cosmo params, B15.3.P2
    #     'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
    #     'MassDef':'200c',  # Mass definition, B15.T2
    #     'MassFunc': 'Tinker08',
    # }
        
    pass


class HaloModel():
    pass


def Fig5(width=14, height=6):
    return BasePlots2(thispath).plot(filename=['Fig5a','Fig5b'], nrow=1, ncol=2, width=width, height=height,
        xlabel=r'$x=r/R_{200}$', ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$',
        xlim=(7e-2, 4), ylim=(1e1, 1e2), xscale='log', yscale='log')