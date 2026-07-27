"""
Dark Energy Survey Year 1 Results: Cosmological Constraints from Cluster Abundances, Weak Lensing, and Galaxy Correlations

ui.adsabs.harvard.edu/abs/2021PhRvL.126n1301T
arxiv.org/pdf/2010.01138
"""


from config import *
class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvL.126n1301T
    subs = {}
    info = {
        }


class SHMRs(BaseSHMR, Studies.To2020):  # DES Y1 Clusters
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
