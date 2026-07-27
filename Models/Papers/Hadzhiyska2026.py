"""
Probing cosmic velocities with the pairwise kinematic Sunyaev-Zel'dovich signal in DESI Bright Galaxy Sample DR1 and ACT DR6

ui.adsabs.harvard.edu/abs/2026PhRvD.113f3565H
arxiv.org/pdf/2510.14135
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Hadzhiyska2026")

from scipy.special import erf
from Models.Papers import Hadzhiyska2025B

class Cosmology():
    pass

class HaloModel():
    pass

class HOD_new(Hadzhiyska2025B.HOD_new):
    # The halo mass definition adopted in the AbacusSummitsimulations is the ‘virial mass’ one from Ref. [128].
    MassDef = 'vir'
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(subsample=self.subsample).to_dict()
        except: self.p0 = {}
    
    # In brief, we use a vanilla Halo Occupation Distribution (HOD) model applied to the AbacusSummit simulations at fixed redshift z = 0.3 [Maksimova2021], appropriate for BGS. The HOD is described by five free parameters, sampled using a Latin Hypercube with 1000 realizations as in the initial work. We adopt the AbacusHOD prescription within theabacusutils package 1. 
    
    
# TABLE I. Summary of the BGS stellar-mass threshold samples, including values of the 5 HOD parameters, inferred satellite fraction, mean halo mass, linear bias, and χ2 null with 4 degrees of freedom. Error bars denote 1σ uncertainties. As estimated in Ref. [123], the systematic and model errors expected from this type of analysis are ∼7%.
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        

# FIG. 2. Top panel: Number of BGS galaxies, with a stellar mass threshold of log10(M⋆/M⊙) > 10, in Y1 as a function of stellar mass in differential bins. The distribution peaks near log10(M⋆/M⊙) ≈ 10.75, reflecting the stellar-mass completeness of the sample at the median survey redshift.
# Bottom panel: Number of BGS galaxies per redshift bin for six cumulative stellar-mass thresholds. Higher-mass samples trace progressively higher-redshift populations, with the mean redshift increasing from  ̄z ≈ 0.25 at log10(M⋆/M⊙) > 10 to  ̄z ≈ 0.36 at log10(M⋆/M⊙) > 11.25. These trends highlight the tradeoff between number density and redshift reach when selecting stellar-mass subsamples. The number of galaxies in each mass bin is given in Table II.
class Fig2(BasePlots2):
    subplots = [
        [dict(name='Fig2a', filename='Fig2a', figsize=(5, 4),
              xlabel=r'$\log (M/M_\odot)$', xlim=(9.9, 12.15), xscale='linear',
              ylabel=r'$N_\text{gal}$', ylim=(-0.6e3, 12.7e3), yscale='linear')],
        [dict(name='Fig2b', filename='Fig2b', figsize=(5, 4),
              xlabel=r'$z$', xlim=(0.06, 0.6), xscale='linear',
              ylabel=r'$N_\text{gal}$', ylim=(-0.5e3, 11.25e3), yscale='linear')],
    ]

    def __init__(self):
        super().__init__(thispath)



"""Old implementation being phased out"""


from Models.Studies import BaseStudy, cycle, Planck2018
class Study(BaseStudy):  # Probing cosmic velocities with the pairwise kinematic Sunyaev-Zel'dovich signal in DESI Bright Galaxy Sample DR1 and ACT DR6, ui.adsabs.harvard.edu/abs/2025arXiv251014135H
    subs = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
    }

    info = Planck2018.info | { # says it uses Planck2018 cosmology
        "MassDef": 'vir',
        "area": 12200*u.deg**2,  # NOTE: area value not given, assumed
        "h": 0.7,  # NOTE: h value not paper, assumed
        "logMsMin": {"M1": 11.861, "M2": 11.918, "M3": 11.831, "M4": 11.750, "M5": 11.947, "M6": 12.183},
        "fsat": {"M1": 0.123, "M2": 0.147, "M3": 0.206, "M4": 0.295, "M5": 0.333, "M6": 0.343},
        "logMhMean": {"M1": 13.135, "M2": 13.184, "M3": 13.266, "M4": 13.310, "M5": 13.370, "M6": 13.456},
        "blin": {"M1": 1.155, "M2": 1.190,  "M3": 1.258, "M4": 1.314, "M5": 1.384, "M6": 1.475},
        
    }
    
    
from Models.HODs import BaseHOD
from Models.Papers import Zheng2005
class HOD(BaseHOD, Study):  #
    models = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']}

    params = {
        "logMcut": {"M1": 11.861, "M2": 11.918, "M3": 11.831, "M4": 11.750, "M5": 11.947, "M6": 12.183},  # characteristic halo mass at which a halo has a 50% probability of hosting a central galaxy
        "logM1": {"M1": 13.257, "M2": 13.186, "M3": 13.024, "M4": 12.802, "M5": 12.769, "M6": 12.823},  # typical halo mass required to host one satellite galaxy
        "sigma_logM": {"M1": 0.534,  "M2": 0.536,  "M3": 0.516,  "M4": 0.524,  "M5": 0.550,  "M6": 0.560},  # scatter in log M describing the smooth transition of the central galaxy occupation function
        "alpha": {"M1": 1.210,  "M2": 1.210,  "M3": 1.322,  "M4": 1.358,  "M5": 1.315,  "M6": 1.346},  # power-law slope governing the number of satellites in high-mass halos
        "kappa": {"M1": 1.254,  "M2": 1.194,  "M3": 1.208,  "M4": 1.251,  "M5": 1.244,  "M6": 1.268}, # times Mcut, cutoff mass below which no satellites are hosted
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 36
        func = lambda p: Zheng2005.HOD().Nc(logM-np.log10(self.h), logMmin=p['logMcut'], sigmalogM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005.HOD().Ns(10**logM/self.h, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)


from Models.TargetData import BaseTargetData
class TargetData(BaseTargetData, Study):  # DESI BGS BRIGHT DR1 stacked with ACT DR6
    path = f"{DATA_PATH}/Hadzhiyska2026"  # location of data, provided by Jenna

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        file = "BGS_BRIGHT_full_noveto_vac_marvin_BGS_ACT_DR6_fixedTh2.10_delta_Ts"
        self.dfdata = pd.read_csv(f"{self.path}/{file}.csv")

    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None, logMsMin=None):
        zscat = self.dfdata.z if logMsMin is None else self.dfdata[self.dfdata.logm>logMsMin].z  # for recreating the plot
        self.catdist('z', zscat, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)

    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None):
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMs', self.dfdata['logm'], qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)

    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None):
        from Models.Papers.Gao2023 import SHMR as Gao2023SHMR
        catlogMhs = Gao2023SHMR(model='Auto').HSMR(self.dfdata['logm'])()

        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMh', catlogMhs, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)