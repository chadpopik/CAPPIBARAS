"""
Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect


ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
arxiv.org/pdf/1207.4061
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))




class ParamsTable(ParamTable):
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = pd.read_csv(filename, dtype={'fixedp': str})





class Fig4(BasePlots2):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(7, 6),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
        dict(name='Fig4b', filename='Fig4b', figsize=(7, 6),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig6(BasePlots2):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subplots = [[
        dict(name='Fig6', filename='Fig6', figsize=(6, 5),
             xlabel=r'$R/R_{500}$', xlim=(0.01, 10), xscale='log',
             ylabel=r'$P/P_{500}/</(M)>$', ylim=(1e-3, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)
