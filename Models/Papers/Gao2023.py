"""
The DESI One-Percent Survey: Constructing Galaxy-Halo Connections for ELGs and LRGs Using Auto and Cross Correlations

ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
arxiv.org/pdf/2306.06317
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    # fixed cosmo info
    h = 0.71
    Om0 = 0.268
    Ol0 = 0.732


class HaloModel():
    MassDef = 'vir'  # Current Virial Mass


class SHMR():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        try: self.p0 = Table3().getparams(model=self.model).to_dict()
        except: self.p0 = {}

    # Section 4. Double power-law form fit to the SHMR for all three Psat models (Table 3).
    def log10Mstar(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        Mh, M0, k = 10**logMh, 10**p['log10M0'], 10**p['log10k']

        Mstar = 2*k / ((Mh/M0)**(-p['beta']) + (Mh/M0)**(-p['alpha']))
        return np.log10(Mstar)


# TABLE 3. Best-fit parameters of the SHMRs for different Psat models.
class Table3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class Fig7a(BasePlots2):
    subplots = [[
        dict(name='Fig7a', filename='Fig7a', figsize=(8, 6),
             xlabel=r'$\log(M_h) \ [M_\odot /h]$', xlim=(10, 15), xscale='linear',
             ylabel=r'$\log(M_*) \ [M_\odot]$', ylim=(7.5, 12), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(9, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(7, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class ParamsTable(ParamTable):  # best-fit SHMR parameters, Table 3
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)
