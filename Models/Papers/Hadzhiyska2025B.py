"""
Missing baryons recovered: A measurement of the gas fraction in galaxies and groups with the kinematic Sunyaev-Zel'dovich effect and CMB lensing

ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
arxiv.org/pdf/2507.14136
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Hadzhiyska2025B")

from scipy.special import erf


    
def Cosmology():
    # = III.A As this analysis is performed at fixed cosmology, we employ the fiducial cosmology boxes which have cosmological parameters set to their Planck 2018 values: Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0.
    # info = {'h': 0.6736, 'MassDef': 'vir'
    # }
    # #Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0
    pass

def HaloModel():
    # III.A. All halo masses quoted in this work correspond to the mass definition adopted by the AbacusSummit halo finder CompaSO [46], which defines the virial mass using the spherical collapse model and the fitting formulae from Bryan and Norman [47].
    MassDef = 'vir'

# III.B. To model the distribution of Luminous Red Galaxies (LRGs) within dark matter halos, we adopt a standard (‘vanilla’) fiveparameter Halo Occupation Distribution (HOD) framework [Zheng et al 2005]. n this model, the mean number of central and satellite galaxies in a halo of mass 𝑀 is given by:
class HOD_new(): 
    MassDef = 'vir'
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1_2_3().getparams(Sample=self.Sample,Bin=self.Bin).to_dict()
        except: self.p0 = {}

    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logM-p['logMcut'])/(2*p['sigmalogM'])))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=p['kappa']*10**p['logMcut'], ((10**self.logM-p['kappa']*10**p['logMcut'])/10**p['logM1']), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)

    
# TABLE I. Best-fit values and 68% confidence intervals for the five HOD parameters and three derived parameters: comoving number density  ̄𝑛 (in [Mpc/ℎ]−3), satellite fraction 𝑓sat, and mean halo mass⟨𝑀halo⟩ (in 𝑀⊙ /ℎ). Results are shown for each of the three tracer samples: Main LRGs, Extended LRGs, and BGS. All mass units are in 𝑀⊙ /ℎ. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 7% for the systematic bias on the mean halo mass, as described in the main text.
# TABLE II. Best-fit HOD and derived parameters for the Main LRG sample, shown across four redshift bins. The redshift bins correspond to: Bin 1: 0.4, 0.54, 0.713, 0.86, 𝑧1 < 𝑧 < 0.54, Bin 2: 0.54 < 𝑧 <0.713, Bin 3: 0.713 < 𝑧 < 0.86, Bin 4: 0.86 < 𝑧 < 1.024. All mass units are in 𝑀⊙ /ℎ, and comoving number densities are in [Mpc/ℎ]−3. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 5% for the systematic bias on the mean halo mass (see Section III D).
# TABLE III. Same as Table II, but for the Extended LRG sample. The redshift bins are identical to those used in Table II.
class Table1_2_3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1_2_3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class ParamsTable(ParamTable):  # characteristic HOD parameters, per mass bin
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)



"""Old implementation being phased out"""


from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # Missing baryons recovered: a measurement of the gas fraction in galaxies and groups with thekinematic Sunyaev-Zel’dovich effect and CMB lensing, ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
    subs = {
    }

    info = {'h': 0.6736, 'MassDef': 'vir'
    }
    #Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0
    
from Models.HODs import BaseHOD
from Models.Papers import Zheng2005
class HOD(BaseHOD, Study):  #
    models = {'sample': ['Main_z1', 'Main_z2', 'Main_z3', 'Main_z4', 'Main_all', 'Ext_z1', 'Ext_z2', 'Ext_z3', 'Ext_z4', 'Ext_all', 'BGS'],
    }
    params = {
        # best fit HOD parameters
        "logMcut": {"Main_z1": 12.61, "Main_z2": 12.63, "Main_z3": 12.73, "Main_z4": 12.63, "Main_all": 12.683, "Ext_z1": 12.49, "Ext_z2": 12.44, "Ext_z3": 12.52, "Ext_z4": 12.38, "Ext_all": 12.491, "BGS": 12.133},  # the characteristic halo mass at which a halo has a 50% probability of hosting a central galaxy, Msol/h
        "logM1": {"Main_z1": 13.91, "Main_z2": 14.02, "Main_z3": 13.98, "Main_z4": 13.97, "Main_all": 14.063, "Ext_z1": 13.96, "Ext_z2": 14.01, "Ext_z3": 14.05, "Ext_z4": 14.10, "Ext_all": 14.196, "BGS": 13.83},  # the typical halo mass required to host one satellite galaxy, Msol/h
        "sigma_logM": {"Main_z1": 0.30, "Main_z2": 0.18, "Main_z3": 0.21, "Main_z4": 0.23, "Main_all": 0.133, "Ext_z1": 0.24, "Ext_z2": 0.14, "Ext_z3": 0.17, "Ext_z4": 0.20, "Ext_all": 0.108, "BGS": 0.107},  # the scatter in log 𝑀 describing the smooth transition of the central galaxy occupation function
        "alpha": {"Main_z1": 1.00, "Main_z2": 0.84, "Main_z3": 0.96, "Main_z4": 0.96, "Main_all": 0.848, "Ext_z1": 0.92, "Ext_z2": 0.76, "Ext_z3": 0.89, "Ext_z4": 0.82, "Ext_all": 0.642, "BGS": 1.219},  # the power-law slope governing the number of satellites in high-mass halos
        "kappa": {"Main_z1": 1.30, "Main_z2": 1.30, "Main_z3": 1.33, "Main_z4": 1.28, "Main_all": 1.245, "Ext_z1": 1.28, "Ext_z2": 1.23, "Ext_z3": 1.23, "Ext_z4": 1.10, "Ext_all": 0.941, "BGS": 1.069},  # multiplied by Mcut, the cutoff mass below which no satellites arehosted
        "nbar_x1000": {"Main_z1": 1.30, "Main_z2": 1.09, "Main_z3": 0.69, "Main_z4": 0.91, "Main_all": 0.745, "Ext_z1": 1.58, "Ext_z2": 1.66, "Ext_z3": 1.16, "Ext_z4": 1.72, "Ext_all": 1.254, "BGS": 3.608},  # comoving number density  ̄𝑛 (in [Mpc/ℎ]−3)
        "fsat": {"Main_z1": 0.08, "Main_z2": 0.09, "Main_z3": 0.08, "Main_z4": 0.07, "Main_all": 0.080, "Ext_z1": 0.07, "Ext_z2": 0.08, "Ext_z3": 0.06, "Ext_z4": 0.06, "Ext_all": 0.098, "BGS": 0.108},  # satellite fraction 𝑓sat,
        "logMh_bar": {"Main_z1": 13.19, "Main_z2": 13.22, "Main_z3": 13.20, "Main_z4": 13.12, "Main_all": 13.179, "Ext_z1": 13.11, "Ext_z2": 13.09, "Ext_z3": 13.05, "Ext_z4": 12.92, "Ext_all": 13.025, "BGS": 13.022},  # mean halo mass⟨𝑀halo⟩ (in 𝑀⊙ /ℎ).
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 2
        logM = logM-np.log10(self.h)
        func = lambda p: Zheng2005.HOD().Nc(logM, logMmin=p['logMcut'], sigmalogM=2*p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 3
        logM = logM-np.log10(self.h)
        func = lambda p: Zheng2005.HOD().Ns(logM, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)