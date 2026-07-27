"""
The Weak Lensing Signal and the Clustering of BOSS Galaxies. II. Astrophysical and Cosmological Constraints

ui.adsabs.harvard.edu/abs/2015ApJ...806....2M
arxiv.org/pdf/1407.1856
"""


from config import *

from scipy.special import erf

from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
    
class HaloModel():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
    
class HOD():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(Sample=self.Sample).to_dict()
        except: self.p0 = {}
    
    def Ncen(self, pdict={}, **kwargs): # Eq 3
        p = self.p0 | pdict | kwargs
        return self.finc(pdict, **kwargs) * (1/2) * (1+erf((self.logM-p['logMmin'])/p['sigmalogM']))

    def Nsat(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=p['kappa']*10**p['logMmin'], ((10**self.logM-p['kappa']*10**p['logMmin'])/10**p['logM1']), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)
    
    def finc(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return np.clip((1+p['alphainc']*(self.logM-p['logMinc'])), 0, 1)
    
    
# The three columns list the 68% confidence intervals on the model parameters for the three stellar mass subsamples we use in our analysis. The parameter M∗,11 denotes the stellar mass in units of 1011 h−2 M and the 68% limit we quote is a one-sided upper limit.
class Table1(ParamTable):  
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        dfraw = read_wide_table(filename)
        self.df, self.df_errup, self.df_errdown = splittable(dfraw)
    

# Figure 2. An illustration of the halo occupation distribution model we use in the analysis of this paper. The red and blue dot-dashed lines show the central and satellite components of the HOD appropriate for a true stellar mass threshold sample. The green dotted line shows the log-linear functional form we assume for parametrizing the incompleteness in our subsample. The solid red, blue and black lines show the HOD of centrals, satellites and all galaxies after accounting for the incompleteness. In total our HOD model is parametrized by 7 parameters.
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 6),
             xlabel=r'$M\ [h^{-1} M_\odot]$', xlim=(1e12, 3.1e15), xscale='log',
             ylabel=r'$\langle N\rangle_M$', ylim=(1e-1, 1e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)

# Figure 4. The 68 and 95 percent confidence intervals of the halo occupation distribution of CMASS galaxies in samples A, B and C obtained from our modeling exercise are shown in the three different panels, respectively. The HOD constraints displayed here are marginalized over the uncertainty in the cosmological parameters. Subsamples A, B and C occupy progressively more massive halos. The results for subsample A can be contrasted with results from White et al. (2011) who constrain the HOD using the clustering of an early data release of CMASS galaxies (shown as solid blue line in left hand panel). The halo occupation distribution for sample C is consistent with the HOD obtained by Reid & Spergel (2009, shown as green solid line in the right hand panel), based on a counts-incylinder analysis of the LRG sample of galaxies. The green dashed line in the right hand panel shows the result of a simple attempt to correct for the differences in the mean redshift of LRGs and CMASS galaxies by adjusting the masses of LRGs at z = 0.3 to the masses of their progenitors at z = 0.53. The gray shaded bands in each of the subsample show the constraints on the HOD obtained by Leauthaud et al. (2012) from COSMOS data, employing the same stellar mass cuts in the galaxy selection. The similarity of our HOD constraints, especially for the higher stellar mass threshold samples, implies that the magnitude of potential incompleteness effects in our analysis decrease with increasing stellar mass threshold.
class Fig4(BasePlots2):
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(4, 4),
             xlabel=r'$M \ [h^{-1}M_\odot]$', xlim=(3.2e11, 3.1e15), xscale='log',
             ylabel=r'$\langle N\rangle_M$', ylim=(1e-2, 2.8e2), yscale='log'),
        dict(name='Fig4b', filename='Fig4b', figsize=(4, 4),
             xlabel=r'$M \ [h^{-1}M_\odot]$', xlim=(3.2e11, 3.1e15), xscale='log',
             ylabel=r'$\langle N\rangle_M$', ylim=(1e-2, 2.8e2), yscale='log'),
        dict(name='Fig4c', filename='Fig4c', figsize=(4, 4),
             xlabel=r'$M \ [h^{-1}M_\odot]$', xlim=(3.2e11, 3.1e15), xscale='log',
             ylabel=r'$\langle N\rangle_M$', ylim=(1e-2, 2.8e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # Free cosmological/model parameters + sample info, Table 1 & Section 2p2, per mbin
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)






class HODParamsTable(ParamTable):  # Best-fit parameters, Table 1
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        self.df = read_wide_table(filename)


