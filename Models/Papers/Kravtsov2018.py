"""
Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos

ui.adsabs.harvard.edu/abs/2018AstL...44....8K
arxiv.org/pdf/1401.7329
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Kravtsov2018")


class Cosmology():
    # Unless otherwise noted, throughout this paper we assume a flat ΛCDM model with parameters Ωm = 1 − ΩΛ = 0.27,Ωb = 0.0469, h = H0/(100 km s−1Mpc−1) = 0.7, σ8 = 0.82 and ns = 0.95 compatible with combined constraints from WMAP, BAO, SNe, and cluster abundance (Vikhlinin et al. 2009b; Komatsu et al. 2011; Hinshaw et al. 2013).
    Om0=0.27
    Ob0=0.0469
    h=0.7
    sigma8=0.82
    ns=0.95
    

class HaloModel():
    # Total masses are defined within radius enclosing a particular overdensity (500 or 200) with respect to the critical density at redshift of observation, which is indicated by a corresponding subscript (M500 or M200).
    pass


class SHMR_new():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

    # Appendix A We adopt the five-parameter parametrization of Behroozi et al. (2013b, see their eq. 3):
    def log10Mstar_B13(self, pdict={}, **kwargs): # Eq A3 A4
        p = self.p0 | pdict | kwargs
        
        f = lambda x : -np.log10(10**([p['alpha']]*x)+1) + p['delta']*(np.log10(1+np.exp(x)))**p['gamma']/(1+np.exp(10**(-x)))
        return np.log10(p['eps']*p['M1']) + f(np.log10(self.Mh/p['M1'])) - f(0)

    # Table II The relations are fit by the power law y = mx + c, wherex = log10 M500 − 14.5 and y is log10 M∗.
    def log10Mstar_PL(self, pdict={}, **kwargs): # Double Power Law
        p = self.p0 | pdict | kwargs
        x = self.logM500-14.5
        return p['slope']*x+p['normalization']


class Table3(ParamTable):
    # TABLE 3 Parameters of best fit M∗ − M parametrization at z . 0.1
    def __init__(self, filename=f"{thispath}/table3.csv"):
        super().__init__(filename)


class Table2(ParamTable):
    # TABLE 2. Best fit parameters for power law fits, y = slope*x + normalization where x = log10(M500)-14.5 and y = log10(M*), for the BCG, satellite, and total stellar mass - halo mass relations, using either the 9 clusters from this work alone or the combined 21-cluster sample (this work + G13).
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class Fig4(BasePlots2):
    subplots = [[
        dict(name='Fig4', filename='Fig4', figsize=(4, 4),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(1.8e13, 2e15), xscale='log',
             ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$', ylim=(5e10, 2.5e13), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig7(BasePlots2):
    subplots = [[
        dict(name='Fig7', filename='Fig7', figsize=(6, 6),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(3.2e13, 2e15), xscale='log',
             ylabel=r'$M_{*, \text{tot}}(<r_{500}) \ [M_\odot]$', ylim=(3.1e11, 6.2e13), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8', filename='Fig8', figsize=(6, 6),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(3.2e13, 2e15), xscale='log',
             ylabel=r'$M_{*, \text{sat}}(<r_{500}) \ [M_\odot]$', ylim=(3.1e11, 6.2e13), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig9(BasePlots2):
    subplots = [[
        dict(name='Fig9', filename='Fig9', figsize=(6, 6),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(2e13, 2e15), xscale='log',
             ylabel=r'$M_{*, \text{BCG}}/(M_{*, \text{BCG}}+M_{*, \text{sat}})$', ylim=(0, 1), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig10(BasePlots2):
    subplots = [[
        dict(name='Fig10', filename='Fig10', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e10, 4e15), xscale='log',
             ylabel=r'$M_{*, \text{cen}} \ [M_\odot]$', ylim=(1e8, 2e13), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig11(BasePlots2):
    subplots = [[
        dict(name='Fig11', filename='Fig11', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e10, 4e15), xscale='log',
             ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$', ylim=(1e-3, 1.55), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig12(BasePlots2):
    subplots = [[
        dict(name='Fig12', filename='Fig12', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e10, 4e15), xscale='log',
             ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$', ylim=(3.1e-3, 1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig13(BasePlots2):
    subplots = [[
        dict(name='Fig13', filename='Fig13', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$', ylim=(1e-3, 1.55), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig14(BasePlots2):
    subplots = [[
        dict(name='Fig14', filename='Fig14', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$', ylim=(3.1e-3, 1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig15(BasePlots2):
    subplots = [[
        dict(name='Fig15a', filename='Fig15a', figsize=(5, 5),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(1.3e11, 1.6e15), xscale='log',
             ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$', ylim=(3e-3, 1), yscale='log'),
        dict(name='Fig15b', filename='Fig15b', figsize=(5, 5),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(1.3e11, 1.6e15), xscale='log',
             ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$', ylim=(3e-3, 1), yscale='log'),
        dict(name='Fig15c', filename='Fig15c', figsize=(5, 5),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(1.3e11, 1.6e15), xscale='log',
             ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$', ylim=(3e-3, 1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig16(BasePlots2):
    subplots = [[
        dict(name='Fig16', filename='Fig16', figsize=(6, 6),
             xlabel=r'$M_{500} \ [M_\odot]$', xlim=(1.8e13, 2e15), xscale='log',
             ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$', ylim=(2.5e10, 2.5e13), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig17(BasePlots2):
    subplots = [[
        dict(name='Fig17', filename='Fig17', figsize=(6, 6),
             xlabel=r'$M_{200} \ [M_\odot]$', xlim=(1e9, 2e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e7, 5e12), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)



class SHMR_B13_Params(ParamTable):  # best fit SHMR params, Table 3
    def __init__(self, filename=f"{thispath}/shmr_b13_params.csv"):
        super().__init__(filename)

class SHMR_PL_Params(ParamTable):  # best fit SHMR params, Table 3
    def __init__(self, filename=f"{thispath}/shmr_pl_params.csv"):
        super().__init__(filename)





"""Old implementation being phased out"""

from Models.Studies import BaseStudy
class Study(BaseStudy):  # Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos, ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    subs = {}
    info = {
        # fixed cosmo params, Section 1pLast
        'Om0':0.27, 'Ob0':0.0469, 'h':0.7, 'sigma8': 0.82, 'ns':0.95,
    }
    
from Models.SHMRs import BaseSHMR
class SHMR(BaseSHMR, Study):  # SDSS DR8 & G13 
    models = {
            'model': ['B13', 'PL'],
            'type': ['BGC', 'sat', 'tot'],
            'data': ['K18', 'K18G13'],
            'mdef': ["200c", "500c", "200m", "vir"],
            'scatter': ['B', 'S']}
    params={
        # best fit SHMR params, Table 3
        "logM1": {
            "B": {"200c": 11.39, "500c": 11.32, "200m": 11.45, "vir": 11.43},
            "S": {"200c": 11.35, "500c": 11.28, "200m": 11.41, "vir": 11.39},},
        "logeps": {
            "B": {"200c": -1.618, "500c": -1.527, "200m": -1.702, "vir": -1.663},
            "S": {"200c": -1.642, "500c": -1.556, "200m": -1.720, "vir": -1.685},},
        "alpha": {
            "B": {"200c": 1.795, "500c": 1.856, "200m": 1.736, "vir": 1.750},
            "S": {"200c": 1.779, "500c": 1.835, "200m": 1.727, "vir": 1.740},},
        "delta": {
            "B": {"200c": 4.345, "500c": 4.376, "200m": 4.273, "vir": 4.290},
            "S": {"200c": 4.394, "500c": 4.437, "200m": 4.305, "vir": 4.335},},
        "gamma": {
            "B": {"200c": 0.619, "500c": 0.644, "200m": 0.613, "vir": 0.595},
            "S": {"200c": 0.547, "500c": 0.567, "200m": 0.544, "vir": 0.531},},
        "slope": {
            "BCG": {"K18": 0.39, "K18G13": 0.33},
            "sat": {"K18": 0.87, "K18G13": 0.75},
            "tot": {"K18": 0.69, "K18G13": 0.59}},
        "norm": {
            "BCG": {"K18": 12.15, "K18G13": 12.24},
            "sat": {"K18": 12.42, "K18G13": 12.52},
            "tot": {"K18": 12.63, "K18G13": 12.71}},
        "scat": {
            "BCG": {"K18": 0.21, "K18G13": 0.17},
            "sat": {"K18": 0.10, "K18G13": 0.10},
            "tot": {"K18": 0.09, "K18G13": 0.11}},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        self.require(['model'])
        if self.model=='B13': self.SHMR = self.SHMR_B13
        elif self.model=='PL': self.SHMR = self.SHMR_PL
        
    def SHMR_PL(self, logMh):  # Eq A3/A4
        self.require(['type', 'data'])
        func = lambda p: p['slope']*(logMh-14.5)-p['norm']
        return lambda p={}: func(self.p0 | p)

    def SHMR_B13(self, logMh):  # Eq A3/A4
        self.require(['mdef', 'scatter'])
        func = lambda p: self.Behroozi(logMh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
        return lambda p={}: func(self.p0 | p)