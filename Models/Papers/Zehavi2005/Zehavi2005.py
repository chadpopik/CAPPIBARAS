""" 
The Luminosity and Color Dependence of the Galaxy Correlation Function

ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z
arxiv.org/pdf/astro-ph/0408569
"""


from config import *

from Models.Papers.PlotsTables import BasePlots2
thispath = os.path.dirname(os.path.abspath(__file__))


class Table3():  # Best-fit HOD Parameters for Luminosity-Threshold Samples
    # Mass is in unit of h−1 Msol
    def __init__(self):
        self.df = pd.read_csv(f"{thispath}/Table3.csv").dropna(how="all")
        
    def getcol(self, key):
        return self.df[key].values

    def getparams(self, kdict={}, **keys):
        df = self.df.copy()
        for k,v in (kdict | keys).items():
            if k not in df.columns: 
                print(f"key {k} not in {df.columns.to_list()}")
                pass
            elif v in df[k].values: df = df.set_index(k).loc[v]
            else: print(f"Value {v} not in {np.unique(df[k].values)}")
        return df


class Cosmology():
    Om=0.3  # Section 4.1


# Section 4.1
class HOD(Cosmology):
    MassDef = "virial"
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        try: self.p0 = Table3().getparams(Mrmax=self.Mrmax).to_dict() 
        except: self.p0 = {}

    # the mean occupation number for central substructurescan be modeled as a step function, i.e., 〈Ncen〉 = 1 for halos with mass M ≥ Mmin and〈Ncen〉 = 0 for M < Mmin
    def Ncen(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return np.heaviside(self.logM - p['logMmin'], 1)

    # the distribution of satellite substructures can be well approximated by a Poisson distribution with the mean following a power-law,〈Nsat〉 = (M/M1)^α, with α ≈ 1.
    def Nsat(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return (10**self.logM/10**p['logM1'])**p['alpha']

    # For luminosity-threshold samples, one galaxy is always assumed to reside at the center of a halo.
    def ncen(self, pdict={}, **kwargs):
        return 1
    
    # In this paper, we assume that the satellite galaxy distribution follows the dark matter distribution within the halo, which we describe by a spherically symmetric NFW profile truncated at the virial radius (defined to enclose a mean overdensity of 200).
    def nsat(self, pdict={}, **kwargs):
        return 1
    
    
# HOD fits to the projected correlation function of the Mr < −20 sample, which has the greatest sensitivity to limiting redshift (Fig. 9). In the left panel, open circles show the measuredwp(rp) for zmax = 0.10, and the dashed curve shows the predic
class Fig15(BasePlots2):
    subplots = [[
        dict(name='Fig15b', filename='Fig15b', figsize=(3, 3),
             xlabel=r'$M (h^{-1} M_\odot)$', xlim=(1e11, 3e15), xscale='log',
             ylabel=r'$\langle N \rangle$', ylim=(0.1, 2e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)
    

class Studies(BaseStudy):  # The Luminosity and Color Dependence of the Galaxy Correlation Function, ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z
    subs = {}
    info = {
        }


class HODs(BaseHOD, Studies.Zehavi2005):  # virial
    models = {}
    params = {}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Nc(self, logM, logMmin):
        return np.heaviside(logM - logMmin, 1)

    def Ns(self, M, M1, alpha):
        return (M/M1)**alpha
