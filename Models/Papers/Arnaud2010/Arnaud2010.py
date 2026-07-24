"""
The universal galaxy cluster pressure profile from a representative sample of nearby systems (REXCESS) and the YSZ - M500 relation

ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
arxiv.org/pdf/0910.1234
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    # 1. We adopt a ΛCDM cosmology with H0 = 70 km/s/Mpc,ΩM = 0.3 and ΩΛ = 0.7. 
    H0 = 70
    Om0 = 0.3
    Ol0 = 0.7
    

class HaloModel():
    # 1. Here and in the following, Mδ and Rδ are the total mass and radius corresponding to a density contrast, δ, as compared to ρc(z), the critical density of the universe at the cluster redshift: Mδ = (4π/3)δρc(z)R3δ .M500 corresponds roughly to the virialised portion of clusters, and is traditionally used to define the ’total’ mass.
    MassDef = '500c'
    
    Concentration = 'Constant'
    

# GNFW model of the universal pressure profile (green line). It is derived by fitting the observed average scaled profile in the radial range [0.03–1]R500 , combined with the average simulation profile beyond R500 (red line).
class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8', filename='Fig8', figsize=(6, 5),
             xlabel=r'Radius $(R_{500})$', xlim=(7.5e-3, 5.2), xscale='log',
             ylabel=r'$P/P_{500}$', ylim=(2.5e-4, 3.4e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class ParamsTable(ParamTable):  # Eq 12, best-fit parameter sets
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)