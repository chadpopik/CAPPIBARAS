""""
Planck 2018 results. VI. Cosmological parameters


ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P
arxiv.org/pdf/1807.06209
"""


from config import *
from Models.Papers.Figures.PlotsTables import ParamTable, read_wide_table, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Planck2018")


class Cosmology():
    def __init__(self, inputdict={}, **inputvars):
        inputs = inputdict | inputvars
        Column = inputs.pop('Column', 'TT,TE,EE+lowE+lensing+BAO')  # which Table 2 dataset combination to take the cosmological parameters from
        params = Table2().getparams(Parameter=Column).to_dict()
        
        for key, value in (params | inputs).items(): setattr(self, key, value)


# Table 1. Base-ΛCDM cosmological parameters from Planck TT,TE,EE+lowE+lensing.
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        dfraw = read_wide_table(filename)
        self.df, self.df_errup, self.df_errdown = splittable(dfraw)


# Table 2. Parameter 68% intervals for the base-ΛCDM model from Planck CMB power spectra, in combination with CMB lensing reconstruction and BAO.
class Table2(ParamTable):
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        dfraw = read_wide_table(filename)
        self.df, self.df_errup, self.df_errdown = splittable(dfraw)




"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # Planck 2018 results. VI. Cosmological parameters, ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P
    subs = {}
    info = { # Table 2 TT,TE,EE+lowE+lensing
        'H0': 67.36*u.km/u.s/u.Mpc, 'Ob0h2':0.02237, 'Oc0h2':0.1200,'ns':0.9649, 'sigma8':0.8120, 'Om0':0.3166, 'Ol0':0.6834
        }