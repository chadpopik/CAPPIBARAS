"""
Dark Energy Survey Year 1 Results: Cosmological Constraints from Cluster Abundances, Weak Lensing, and Galaxy Correlations

ui.adsabs.harvard.edu/abs/2021PhRvL.126n1301T
arxiv.org/pdf/2010.01138
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "To2020")

"""Old implementation being phased out"""
from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvL.126n1301T
    subs = {}
    info = {
        }


class SHMRs(Study):  # DES Y1 Clusters
    models = {}
    params = {
        'alpha1': 14.351,
        'alpha2': 1.058,
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
    
    def RHMR(self, richness):
        func = lambda p: 10**p['alpha1'] * (richness/40)**p['alpha2'] / self.h*u.Msun
        return lambda p={}: func(self.p0 | p)
