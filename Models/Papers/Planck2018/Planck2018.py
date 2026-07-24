""""
Planck 2018 results. VI. Cosmological parameters


ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P
arxiv.org/pdf/1807.06209
"""


from config import *
from Models.Papers.PlotsTables import ParamTable, read_wide_table, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


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
