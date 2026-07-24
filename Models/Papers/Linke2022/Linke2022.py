"""
KiDS+VIKING+GAMA: Halo occupation distributions and correlations of satellite numbers with a new halo model of the galaxy-matter bispectrum for galaxy-galaxy-galaxy lensing

ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
arxiv.org/pdf/2204.02418
"""


from config import *

from scipy.special import erf

from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


class HOD():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


    def Ncen(self, pdict={}, **kwargs): # Eq 36
        p = self.p0 | pdict | kwargs
        return (p['alpha_a']/2) * (1+erf((self.logM-p['logMtha'])/p['sigmaa']))


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(8, 6),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'$\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8a', filename='Fig8a', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
        dict(name='Fig8b', filename='Fig8b', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # best fit params + Cosmo params, per sample (color-independent)
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)



class HODParamsTable(ParamTable):  # Best-fit parameters, Table 3
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        super().__init__(filename)

