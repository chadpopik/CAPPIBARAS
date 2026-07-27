"""
Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies

ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
arxiv.org/pdf/astro-ph/0408564
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Zheng2005")


from scipy.special import erf

from Models.Papers import Zehavi2005


class Cosmology():
    pass

class HaloModel():
    pass

class HOD_new(): # Section 3.1. HOD for All Galaxies
    MassDef = "virial"
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(ngbar=self.ngbar,Model=self.Model,numParams=self.numParams).to_dict()
        except: self.p0 = {}
        
        self.Zehavi2005 = Zehavi2005.HOD(inputdict | inputvars)
        
    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        
        if self.numParams=='3P':
            return self.Zehavi2005.Ncen(p)
            
        return (1/2) * (1+erf((self.logM-p['logMmin'])/p['sigmalogM']))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
        
        if self.numParams=='5P':
            return self.Zehavi2005.Nsat(p)
        
        return np.where(10**self.logM>=p['M0'], ((10**self.logM-10**p['M0'])/10**p['logM1prime']), 0)**p['alpha']


# Table 1. HOD Parameters for Galaxy Samples with Different Thresholds of Baryonic Mass
# Note. — Number density and mass are in units of h3Mpc−3 and M⊙, respectively. Columns 3–5 are for the 3-parameter model and Columns 6–10 are for the 5-parameter model (see the text). For the 3-parameter model, Mmin is simply set to be the halo mass at which 〈N 〉M = 0.5, and M1 and α are obtained through a power-law fit to data points with 〈Nsat〉M > 0.1.
class Table1():  
    def __init__(self):
        self.df = pd.read_csv(f"{thispath}/Table1.csv")

    def getcol(self, key):
        return self.df[key].values

    def getparams(self, **keys):
        df = self.df.copy()
        for k,v in keys.items():
            if k not in df.columns: 
                print(f"key {k} not in {df.columns.to_list()}")
                pass
            elif v in df[k].values: df = df.set_index(k).loc[v]
            else: print(f"Value {v} not in {np.unique(df[k].values)}")
        return df

    
#Fig. 1.— Mean occupation number and scatter as a function of halo mass, separated into central and satellite galaxies. Predictions are shown for the  ̄ng = 0.02h3 Mpc−3 samples from the SPH simulation (left panels) and from the SA model (right panels). Lower panels plot the mean occupation numbers of central, satellite, and all galaxies. In the upper panels, circles show 〈N (N − 1)〉1/2/〈N 〉, indicating the width of the probability distribution, for all galaxies (filled circles) and satellite galaxies (open circles). For Poisson P (N |M ), this ratio would be one (dotted line). This figure can be compared to Fig. 4 of K04.
class Fig1(BasePlots2):
    subplots = [[
        dict(name='Fig1c', filename='Fig1c', figsize=(5, 4),
             xlabel=r'$M (M_\odot)$', xlim=(3.15e10, 1e15), xscale='log',
             ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 50), yscale='log'),
        dict(name='Fig1d', filename='Fig1d', figsize=(5, 4),
             xlabel=r'$M (M_\odot)$', xlim=(3.15e10, 1e15), xscale='log',
             ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 50), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Fig. 3.— Parameterized fits to mean occupation functions (top panels) and predicted numbers of galaxy pairs and triplets (middle and bottom panels) for the SPH simulation (left) and the SA model (right). For each model, left panels show results based on 3-parameter fits, which assume sharp cutoff profiles of 〈Ncen〉M and 〈Nsat〉M , and right panels show results of fits with more parameters to model the cutoff profiles (see eqs. [1] and [3]). Fits and predictions are plotted as curves, and circles are measurements from the models.
class Fig3(BasePlots2):
    subplots = [
        [dict(name='Fig3a', filename='Fig3a', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log'),
         dict(name='Fig3b', filename='Fig3b', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log')],
        [dict(name='Fig3c', filename='Fig3c', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log'),
         dict(name='Fig3d', filename='Fig3d', figsize=(4, 3),
              xlabel=r'$M (M_\odot)$', xlim=(1e11, 1e15), xscale='log',
              ylabel=r'$\langle N \rangle$', ylim=(3.1e-2, 4.9e1), yscale='log')],
    ]

    def __init__(self):
        super().__init__(thispath)


class HODParamsTable(ParamTable):  # best-fit HOD parameters, Table 1 (same data as Table1.csv above, kept string-typed for exact key matching)
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        self.df = pd.read_csv(filename, dtype={'ngbar': str})




"""Old implementation being phased out"""



from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies, ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    subs = {}
    info = {'MassDef': 'vir',
        }
    
from Models.HODs import BaseHOD
class HOD(BaseHOD, Study):  # virial mass
    models = {'model': ['SPH', 'SA'],  # simulation type
              'nparams': ['3P', '5P'],  # number of parameters in model (Zehavi vs Zheng)
              'ng': ['0.02', '0.01', '0.005', '0.0025'],  # number density in h^3 Mpc^-3
    }
    params = {  # best-fit HOD parameters, Table 1
        "logMmin": {  # characteristic minimum mass of halos that can host such central galaxies
            "3P": {"SPH": {"0.02": 11.67, "0.01": 12.04, "0.005": 12.38, "0.0025": 12.68},
                   "SA":  {"0.02": 11.77, "0.01": 12.03, "0.005": 12.34, "0.0025": 12.58},},
            "5P": {"SPH": {"0.02": 11.68, "0.01": 12.07, "0.005": 12.36, "0.0025": 12.69},
                   "SA":  {"0.02": 11.73, "0.01": 12.02, "0.005": 12.36, "0.0025": 12.60},},},
        "logM1": {
            "3P": {"SPH": {"0.02": 12.96, "0.01": 13.26, "0.005": 13.55, "0.0025": 13.85},
                   "SA":  {"0.02": 12.96, "0.01": 13.30, "0.005": 13.64, "0.0025": 13.91},},
            "5P": {"SPH": {"0.02": 13.00, "0.01": 13.19, "0.005": 13.45, "0.0025": 13.82},
                   "SA":  {"0.02": 12.87, "0.01": 13.32, "0.005": 13.62, "0.0025": 13.86},},},
        "alpha": {
            "3P": {"SPH": {"0.02": 0.97, "0.01": 1.03, "0.005": 1.18, "0.0025": 1.24},
                   "SA":  {"0.02": 1.04, "0.01": 1.02, "0.005": 1.09, "0.0025": 1.16},},
            "5P": {"SPH": {"0.02": 1.02, "0.01": 0.94, "0.005": 1.00, "0.0025": 1.08},
                   "SA":  {"0.02": 0.96, "0.01": 1.07, "0.005": 1.04, "0.0025": 1.03},},},
        "sigmalogM": { # characteristic transition width
            "5P": {"SPH": {"0.02": 0.15, "0.01": 0.18, "0.005": 0.15, "0.0025": 0.15},
                   "SA":  {"0.02": 0.32, "0.01": 0.26, "0.005": 0.42, "0.0025": 0.28},},},
        "logM0": {  # truncation mass for satellites
            "5P": {"SPH": {"0.02": 11.86, "0.01": 12.28, "0.005": 12.63, "0.0025": 12.94},
                   "SA":  {"0.02": 12.09, "0.01": 12.28, "0.005": 12.28, "0.0025": 12.77},},},
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
    def Nc(self, logM, logMmin, sigmalogM):  # Eq 1
        return (1/2) * (1+erf((logM-logMmin)/sigmalogM))

    def Ns(self, M, M0, M1, alpha):  # Eq 3
        return np.where(M>=M0, ((M-M0)/M1), 0)**alpha

    def Ncen(self, logM):
        if self.nparams=='3P': func = lambda p: Zehavi2005.HOD().Nc(logM, logMmin=p['logMmin'])
        elif self.nparams=='5P': func = lambda p: self.Nc(logM, logMmin=p['logMmin'], sigmalogM=p['sigmalogM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):
        if self.nparams=='3P': func = lambda p: Zehavi2005.HOD().Ns(10**logM, M1=10**p['logM1'], alpha=p['alpha'])
        elif self.nparams=='5P': func = lambda p: self.Ns(10**logM, M0=10**p['logM0'], M1=10**p['logM1'], alpha=p['alpha'])
        return lambda p={}: func(self.p0 | p)