"""
Cosmological constraints from the tomographic cross-correlation of DESI Luminous Red Galaxies and Planck CMB lensing

ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
arxiv.org/pdf/2111.09898
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))




class StudiesInfoTable(ParamTable):  # shot noise, effective z, ell limits, per zbin
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)



class Fig2(BasePlots2):  # ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(8, 4),
             xlabel=r'$z$', xlim=(0.2, 1.2), xscale='linear',
             ylabel=r'$d \text{ln} N/dz$ (actually $\frac{1}{N}\frac{dN}{dz}$)', ylim=(-0.3, 7.05), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)
