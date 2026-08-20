"""
Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect


ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
arxiv.org/pdf/1207.4061
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Planck2013")




class ParamsTable(ParamTable):
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = pd.read_csv(filename, dtype={'fixedp': str})





class Fig4(BasePlots2):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(7, 6),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
        dict(name='Fig4b', filename='Fig4b', figsize=(7, 6),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig6(BasePlots2):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subplots = [[
        dict(name='Fig6', filename='Fig6', figsize=(6, 5),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)






"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect, ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subs = {}

    info = {'Om0':0.3, 'Ol0':0.7,'h':0.7,'MassDef':'500c',
        }
    
from CAPPIBARAS.Models.OldModules.Profiles import BaseProfile
from Models.Papers import Arnaud2010
class Planck2013(BaseProfile, Study): # In progress
    models = {'cluster':['All', 'cool', 'noncool'],
              'fixedp': ['3','2','1','0']}
    params = {
        'P0': {'All':{'3':6.32,'2':6.82,'1':6.41,'0':5.78}, 'cool':11.82, 'noncool':4.72},
        'c500': {'All':{'3':1.01,'2':1.13,'1':1.81,'0':1.84}, 'cool':0.60, 'noncool':2.19},
        'gamma': {'All':{'3':0.31,'2':0.31,'1':0.31,'0':0.35}, 'cool':0.31, 'noncool':0.31},
        'alpha': {'All':{'3':1.05,'2':1.05,'1':1.33,'0':1.39}, 'cool':0.76, 'noncool':1.82},
        'beta': {'All':{'3':5.49,'2':5.17,'1':4.13,'0':4.05}, 'cool':6.58, 'noncool':3.62},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
    def P500(self, z, logM500c, units='cosmo'):
        return Arnaud2010.HaloProfiles(H=self.H).P500(z, logM500c, units)
    
    def PGNFW(self, x, gamma, alpha, beta, P0, c500):
        return Arnaud2010.HaloProfiles().PGNFW(x, gamma, alpha, beta, P0, c500)

    def mdep(self, x, logM500c):
        return Arnaud2010.HaloProfiles().mdep(x, logM500c)({'alpha_P': 0.12})
    
    def Pressure(self, r, z, logM500c, units='cosmo'):
        A10P = Arnaud2010.HaloProfiles(H=self.H, r500c=self.r500c).Pressure(r, z, logM500c, units)
        return lambda p={}: A10P(self.p0 | p)