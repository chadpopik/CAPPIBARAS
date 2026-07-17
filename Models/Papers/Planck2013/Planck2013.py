"""
Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect


ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
arxiv.org/pdf/1207.4061
"""

import os
import pandas as pd
from Models.Papers.PlotsTables import ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))


    # subs = {}

    # info = {'Om0':0.3, 'Ol0':0.7,'h':0.7,'MassDef':'500c',
    #     }
    
    
    #     def Fig4(self, width=14, height=6):
    #     return self.plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
    #         xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
    #         xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')
        
    # def Fig6(self, width=6, height=5):
    #     return self.plot(filename='Fig6', width=width, height=height,
    #         xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
    #         xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')


class Studies(BaseStudy):  # Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect, ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subs = {}

    info = {'Om0':0.3, 'Ol0':0.7,'h':0.7,'MassDef':'500c',
        }


class ParamsTable(ParamTable):
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = pd.read_csv(filename, dtype={'fixedp': str})


class Profiles(BaseProfile, Studies.Planck2013): # In progress
    models = {'cluster':['All', 'cool', 'noncool'],
              'fixedp': ['3','2','1','0']}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = ParamsTable().getparams(cluster=self.cluster, fixedp=self.fixedp).to_dict()
        
    def P500(self, z, logM500c, units='cosmo'):
        return Arnaud2010(H=self.H).P500(z, logM500c, units)
    
    def PGNFW(self, x, gamma, alpha, beta, P0, c500):
        return Arnaud2010().PGNFW(x, gamma, alpha, beta, P0, c500)

    def mdep(self, x, logM500c):
        return Arnaud2010().mdep(x, logM500c)({'alpha_P': 0.12})
    
    def Pressure(self, r, z, logM500c, units='cosmo'):
        A10P = Arnaud2010(H=self.H, r500c=self.r500c).Pressure(r, z, logM500c, units)
        return lambda p={}: A10P(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    def Fig4(self, width=14, height=6):
        return self.plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
            xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')
        
    def Fig6(self, width=6, height=5):
        return self.plot(filename='Fig6', width=width, height=height,
            xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
            xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')
