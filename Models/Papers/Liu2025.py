"""
Measurements of the thermal Sunyaev-Zel'dovich effect with ACT and DESI luminous red galaxies


ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
arxiv.org/pdf/2502.08850
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))





class TargetDataInfoTable(ParamTable):  # rough mean halo mass (Yuan 2023) + T1, per zbin
    def __init__(self, filename=f"{thispath}/target_data_info.csv"):
        self.df = read_wide_table(filename)




class Fig2(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 5),
             xlabel=r'$z$', xlim=(0.1, 1.3), xscale='linear',
             ylabel=r'$dN/dz$ (actually $n(z)$)', ylim=(0, 10.25), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig3(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.8e-6, 4.4e-6), yscale='linear'),
        dict(name='Fig3b', filename='Fig3b', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.8e-6, 4.4e-6), yscale='linear'),
    ], [
        dict(name='Fig3c', filename='Fig3c', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-1.15e-6, 4.2e-6), yscale='linear'),
        dict(name='Fig3d', filename='Fig3d', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-1.15e-6, 4.2e-6), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig4(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.45e-6, 5.5e-6), yscale='linear'),
        dict(name='Fig4b', filename='Fig4b', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.45e-6, 5.5e-6), yscale='linear'),
    ], [
        dict(name='Fig4c', filename='Fig4c', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.5e-6, 5.5e-6), yscale='linear'),
        dict(name='Fig4d', filename='Fig4d', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.5e-6, 5.5e-6), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)
