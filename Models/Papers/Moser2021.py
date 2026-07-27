"""
The Impacts of Modeling Choices on the Inference of Circumgalactic Medium Properties from Sunyaev-Zeldovich Observations

arxiv.org/pdf/2103.02469
ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


class Table3_bestfit(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_bestfit.csv"):
        super().__init__(filename)

class Table3_marginalized(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_marginalized.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)

class Table4(ParamTable):  # best-fit values from 2D fits
    def __init__(self, filename=f"{thispath}/table4.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        

class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(7, 6),
             xlabel=r'$r/r_{200c}$', xlim=(8e-2, 6.2e0), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(5e-31, 1.2e-26), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(7, 6),
             xlabel=r'$r/r_{200c}$', xlim=(8e-2, 6.2e0), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1e-16, 1.6e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig3(BasePlots2):
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(7, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(10.8, 16), xscale='linear',
             ylabel=r'$M_s \ [M_\odot]$', ylim=(7, 12.5), yscale='linear'),
        dict(name='Fig3b', filename='Fig3b', figsize=(7, 6),
             xlabel=r'$\log_10(M^*)\ (M_\odot)$', xlim=(10.6, 11.8), xscale='linear',
             ylabel='', ylim=(2, 5.5e4), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig4row1(BasePlots2):
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(6.5e-31, 4e-26), yscale='log'),
        dict(name='Fig4b', filename='Fig4b', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1.5e-16, 1.1e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig6col1(BasePlots2):
    subplots = [[
        dict(name='Fig6a', filename='Fig6a', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(6.5e-31, 4e-26), yscale='log'),
        dict(name='Fig6c', filename='Fig6c', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1.5e-16, 1.1e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)

