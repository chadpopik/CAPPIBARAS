"""
Photometric Objects Around Cosmic Webs (PAC) Delineated in a Spectroscopic Survey. IV. High-precision Constraints on the Evolution of the Stellar-Halo Mass Relation at Redshift z < 0.7

ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
arxiv.org/pdf/2211.02665
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Xu2023")




class Cosmology():
    # 1. We adopt the cosmology with Ωm = 0.268, ΩΛ = 0.732 andH0 = 71 km/s/Mpc throughout the paper.
    Om0 = 0.268
    Ol0 = 0.732
    sigma8 = 0.831
    h = 0.71


class HaloModel():
    # 3.2 Here we define Macc as the viral mass Mvir of the halo at the time when the galaxy was last the central dominant object.
    MassDef = 'vir'


class SHMR_new():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        try: self.p0 = Table2().getparams(model=self.model, redshift=self.redshift).to_dict()
        except: self.p0 = {}
        
    # 3.2 To parameterize the SHMR, the most commonly used five-parameter formula is a double power law with a scatter (Wang & Jing 2010; Yang et al. 2012; Moster et al. 2013): Eq 8
    def log10Mstar_DP(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        Mh, M0, k = 10**logMh, 10**p['log10M0'], 10**p['log10k_log10eps']

        Mstar = 2*k / ((Mh/M0)**(-p['beta']) + (Mh/M0)**(-p['alpha']))
        return np.log10(Mstar)

    # 3.2. However, Behroozi et al. (2013) found that the SHMR of the double power law form (hereafter DP) fail to reproduce the upturn feature in the GSMF at M∗ <109.5M . They provided a six-parameter formula (here-after BP13) for the SHMR of low mass galaxies: Eq 9
    def log10Mstar_BP13(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        M0, eps = 10**p['log10M0'], 10**p['log10k_log10eps']

        f = lambda x: -np.log10(10**(-p['beta']*x)+1) + p['delta']*(np.log10(1+np.exp(x)))**p['alpha']/(1+np.exp(10**(-x)))
        return np.log10(eps*M0) + f(logMh-p['log10M0']) - f(0)


# Table 2. Posterior PDFs of the parameters from MCMC for the SHMR models.
class Table2(ParamTable):
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        
        
class ParamsTable(ParamTable):  # best-fit SHMR parameters
    def __init__(self, filename=f"{thispath}/params.csv"):
        super().__init__(filename)


# Figure 7. The mean stellar-halo mass relations (lines) and 1σ errors (shadows) at different redshift ranges and from both the BP13 (solid lines) and DP models (dashed lines). The horizontal lines indicate the stellar mass limit covered by the observation data at each redshift matched by color.
class Fig7(BasePlots2):
    subplots = [[
        dict(name='Fig7', filename='Fig7', figsize=(6, 6),
             xlabel=r'$M_h \ [h^{-1} \ M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 9. Comparison of the our mean stellar-halo mass function to previous studies at zs ∼ 0.1. Results compared include those from empirical modeling (Behroozi et al. 2013; Moster et al. 2013), from abundance matching (Guo et al. 2010) and from Conditional Stellar Mass Function (CSMF) modeling (Yang et al. 2012).
class Fig9(BasePlots2):
    subplots = [[
        dict(name='Fig9', filename='Fig9', figsize=(8, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* / M_h$', ylim=(1e-4, 1e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)

# Figure 13. HODs of the LOWZ (left) and CMASS (right) samples derived from our SHMR after taking the stellar mass completeness into account. The HOD of the CMASS sample from Yuan et al. (2022a) are also presented for comparison.
class Fig13(BasePlots2):
    subplots = [[
        dict(name='Fig13a', filename='Fig13a', figsize=(6, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
        dict(name='Fig13b', filename='Fig13b', figsize=(6, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(1e10, 1e15), xscale='log',
             ylabel=r'$M_* \ [M_\odot]$', ylim=(1e6, 1e12), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)







"""Old implementation being phased out"""



from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
    subs = {'sample': ['Main', 'LOWZ', 'CMASS'],
            'form': ['BP13', 'DP']}
    info = {
        # Fixed cosmo parameters, Section 3.1
        'Om0':0.268, 'Ol0':0.732, 'sigma8':0.831, 'h':0.71,
        # Mass definition, Eq 8 
        'mdef': 'vir',  # virial mass of the halo at the time when the galaxy was last the central dominant object
    }
    
from Models.SHMRs import BaseSHMR
class SHMR(BaseSHMR, Study):  # SDSS DR7 Main and SDSSIII BOSS DR12 LOWZ & CMASS
    models = {'sample': ['Main', 'LOWZ', 'CMASS'],  # galaxy sample
            'form': ['BP13', 'DP'],  # form of SHMR
            }
    params = {  # best-fit SHMR parameters
        "logM0": {  # Msun/h
            "BP13": {"Main": 11.338, "LOWZ": 11.359, "CMASS": 11.509},
            "DP":   {"Main": 11.732, "LOWZ": 11.579, "CMASS": 11.624}},
        "alpha": {  # slope of high mass end of SHMR
            "BP13": {"Main": 0.484, "LOWZ": 0.623, "CMASS": 0.740},
            "DP":   {"Main": 0.299, "LOWZ": 0.429, "CMASS": 0.466}},
        "delta": {
            "BP13": {"Main": 3.041, "LOWZ": 3.248, "CMASS": 2.964}},
        "beta": {  # slope of low mass end of SHMR
            "BP13": {"Main": 1.632, "LOWZ": 1.702, "CMASS": 2.094},
            "DP":   {"Main": 1.917, "LOWZ": 2.215, "CMASS": 2.513}},
        "logeps": {  # TODO: check units on this
            "BP13": {"Main": -1.545, "LOWZ": -1.598, "CMASS": -1.565}},
        "logk": {  # TODO: check units on this
            "DP":   {"Main": 10.303, "LOWZ": 10.105, "CMASS": 10.133}},
        "sigma": {  # width of gaussian function that scatter logMs at a given Macc, TODO: implement
            "BP13": {"Main": 0.237, "LOWZ": 0.190, "CMASS": 0.190},
            "DP":   {"Main": 0.233, "LOWZ": 0.201, "CMASS": 0.192}},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def SHMR(self, logMh):
        self.require(['form', 'sample'])
        if self.form=='BP13': return self.SHMR_BP13(logMh)
        elif self.form=='DP': return self.SHMR_DP(logMh)

    def SHMR_BP13(self, logMh):
        func = lambda p: self.Behroozi(logMh-np.log10(self.h), logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        return lambda p={}: func(self.p0 | p)

    def SHMR_DP(self, logMh):
        func = lambda p: self.DoublePowerLaw(logMh-np.log10(self.h), logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)