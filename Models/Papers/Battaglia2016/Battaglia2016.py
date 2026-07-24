"""
The tau of galaxy clusters

ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
arxiv.org/pdf/1607.02442
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    
    
    #     info = {
    #     # sim's cosmo params, B15.3.P2
    #     'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
    #     'MassDef':'200c',  # Mass definition, B15.T2
    #     'MassFunc': 'Tinker08',
    # }
        
    pass


class HaloModel():
    pass


class Fig5(BasePlots2):
    subplots = [[
        dict(name='Fig5a', filename='Fig5a', figsize=(7, 6),
             xlabel=r'$x=r/R_{200}$', xlim=(7e-2, 4), xscale='log',
             ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$', ylim=(1e1, 1e2), yscale='log'),
        dict(name='Fig5b', filename='Fig5b', figsize=(7, 6),
             xlabel=r'$x=r/R_{200}$', xlim=(7e-2, 4), xscale='log',
             ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$', ylim=(1e1, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)



class ParamsTable(ParamTable):  # best-fit GNFW parameters, Table 2
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)

