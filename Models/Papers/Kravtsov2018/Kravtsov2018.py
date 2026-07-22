"""
Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos

ui.adsabs.harvard.edu/abs/2018AstL...44....8K
arxiv.org/pdf/1401.7329
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


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


class SHMR():
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


class Studies(BaseStudy):  # Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos, ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    subs = {}
    info = {
        # fixed cosmo params, Section 1pLast
        'Om0':0.27, 'Ob0':0.0469, 'h':0.7, 'sigma8': 0.82, 'ns':0.95,
    }


class SHMR_B13_Params(ParamTable):  # best fit SHMR params, Table 3
    def __init__(self, filename=f"{thispath}/shmr_b13_params.csv"):
        super().__init__(filename)

class SHMR_PL_Params(ParamTable):  # best fit SHMR params, Table 3
    def __init__(self, filename=f"{thispath}/shmr_pl_params.csv"):
        super().__init__(filename)


class SHMRs(BaseSHMR, Studies.Kravtsov2018):  # SDSS DR8 & G13
    models = {
            'model': ['B13', 'PL'],
            'type': ['BGC', 'sat', 'tot'],
            'data': ['K18', 'K18G13'],
            'mdef': ["200c", "500c", "200m", "vir"],
            'scatter': ['B', 'S']}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.require(['model'])
        if self.model=='B13':
            self.SHMR = self.SHMR_B13
            self.p0 = SHMR_B13_Params().getparams(scatter=self.scatter, mdef=self.mdef).to_dict()
        elif self.model=='PL':
            self.SHMR = self.SHMR_PL
            self.p0 = SHMR_PL_Params().getparams(type=self.type, data=self.data).to_dict()
        
    def SHMR_PL(self, logMh):  # Eq A3/A4
        self.require(['type', 'data'])
        func = lambda p: p['slope']*(logMh-14.5)-p['norm']
        return lambda p={}: func(self.p0 | p)

    def SHMR_B13(self, logMh):  # Eq A3/A4
        self.require(['mdef', 'scatter'])
        func = lambda p: self.Behroozi(logMh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
        return lambda p={}: func(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    def Fig4(self, width=4, height=4):
        return self.plot(filename='Fig4', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
            xlim=(1.8e13, 2e15), ylim=(5e10, 2.5e13), xscale='log', yscale='log')

    def Fig7(self, width=6, height=6):
        return self.plot(filename='Fig7', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}(<r_{500}) \ [M_\odot]$',
            xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

    def Fig8(self, width=6, height=6):
        return self.plot(filename='Fig8', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{sat}}(<r_{500}) \ [M_\odot]$',
            xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

    def Fig9(self, width=6, height=6):
        return self.plot(filename='Fig9', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}}/(M_{*, \text{BCG}}+M_{*, \text{sat}})$',
            xlim=(2e13, 2e15), ylim=(0, 1), xscale='log')

    def Fig10(self, width=6, height=6):
        return self.plot(filename='Fig10', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}} \ [M_\odot]$',
            xlim=(1e10, 4e15), ylim=(1e8, 2e13), xscale='log', yscale='log')

    def Fig11(self, width=6, height=6):
        return self.plot(filename='Fig11', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 4e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

    def Fig12(self, width=6, height=6):
        return self.plot(filename='Fig12', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 4e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

    def Fig13(self, width=6, height=6):
        return self.plot(filename='Fig13', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 1e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

    def Fig14(self, width=6, height=6):
        return self.plot(filename='Fig14', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 1e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

    def Fig15(self, width=15, height=5):
        return self.plot(filename=['Fig15a','Fig15b','Fig15c'], nrow=1, ncol=3, width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$',
            xlim=(1.3e11, 1.6e15), ylim=(3e-3, 1), xscale='log', yscale='log')

    def Fig16(self, width=6, height=6):
        return self.plot(filename='Fig16', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
            xlim=(1.8e13, 2e15), ylim=(2.5e10, 2.5e13), xscale='log', yscale='log')

    def Fig17(self, width=6, height=6):
        return self.plot(filename='Fig17', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_* \ [M_\odot]$',
            xlim=(1e9, 2e15), ylim=(1e7, 5e12), xscale='log', yscale='log')
