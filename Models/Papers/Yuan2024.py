"""
The DESI one-per cent survey: exploring the halo occupation distribution of luminous red galaxies and quasi-stellar objects with ABACUSSUMMIT

ui.adsabs.harvard.edu/abs/2024MNRAS.530..947Y
arxiv.org/pdf/2306.06314
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Yuan2024")

from scipy.special import erf

class Cosmology():
    # 1. Throughout this paper, we adopt the Planck 2018 ΛCDM cosmology, specifically the mean estimates of the Planck TT,TE,EE+lowE+lensing likelihood chains: Ω𝑐 ℎ2 = 0.1200,Ω𝑏 ℎ2 = 0.02237, 𝜎8 = 0.811355, 𝑛𝑠 = 0.9649, ℎ = 0.6736,𝑤0 = −1 and 𝑤𝑎 = 0
    cosmoparams = {
        'Oc0h2': 0.1200,
        'Ob0h2': 0.02237,
        'sigma8': 0.811355,
        'ns': 0.9649,
        'h': 0.6736,
        'w0': -1,
        'wa': 0
    }
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (self.cosmoparams | inputdict | inputvars).items(): setattr(self, key, value)

        self.Omega_b = getattr(self, "Omega_b", self.Ob0h2 / self.h**2)
        self.Omega_c = getattr(self, "Omega_c", self.Oc0h2 / self.h**2)


class HaloModel():
    MassDef = '200c' # NOTE: figure this out
    Concentration = 'Diemer15' # NOTE: figure this out too
    def __init__(self, cosmology=None, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        self.cosmology = cosmology or Cosmology()

        from Models.Codes import pyccl
        self.pyccl = pyccl.HaloModel(            
            Omega_b=self.cosmology.Omega_b,
            Omega_c=self.cosmology.Omega_c,
            h=self.cosmology.h,
            sigma8=self.cosmology.sigma8,
            n_s=self.cosmology.ns,
            MassDef=self.MassDef,
            Concentration = self.Concentration)
         
    def NFW_real(self, r, z, Mh):
        return self.pyccl.NFW_r(r=r, z=z, logM=np.log10(Mh/u.Msun))
    
    def NFW_fourier(self, k, z, Mh):
        return self.pyccl.NFW_k(k=k, z=z, logM=np.log10(Mh/u.Msun), trunc=True, normalized=True)


class HOD():
    MassDef = '??????' # NOTE: figure this out
    def __init__(self, logMh, cosmology=None, halomodel=None, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        self.cosmology = cosmology or Cosmology()
        self.halomodel = halomodel or HaloModel(self.cosmology)

        """Table 1, Table 2: Units of mass are in ℎ−1 𝑀⊙"""
        self.Mh = 10**logMh/self.cosmology.h*u.Msun
        self.logMh = np.log10(self.Mh/u.Msun)
        
        if hasattr(self, "r"):
            from Models.Codes import mcfit
            fft = mcfit.FFT(rs=self.r)  # setup FFT
            self.k, self.FFT3D, self.IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions

        if hasattr(self, "k"):
            self.NFW = self.halomodel.NFW_k(k=self.k, z=self.z, logM=self.logMh)
            self.NFW_norm = self.NFW/self.NFW[0,:, :]
            self.dirac = np.ones_like(self.k)[:, None, None]
        
        table = Table3_4()
        missing = False
        for col in ("Tracer", "Model"):  # validate each against the values in its table column
            options = table.df[col].dropna().unique().tolist()
            value = getattr(self, col, None)
            if value is None:
                print(f"No {col} specified. Valid {col} options: {options}")
                missing = True
            elif value not in options:
                print(f"{col} '{value}' is not a valid option. Valid {col} options: {options}")
                missing = True

        if missing:
            self.p0 = {}
        else:
            try: self.p0 = table.getparams(Tracer=self.Tracer, Model=self.Model).to_dict()
            except: self.p0 = {}
        

    """4.1 Baseline Model"""
    """For a LRG sample, the HOD is well approximated by a vanilla model
    given by (originally shown in Zheng et al. 2007 and referred to as Zheng07 or vanilla later in the text): [Eq 4 & 5]"""
    def Ncen(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return (p['fic']/2) * (1+erf((self.logMh-p['logMcut'])/(np.sqrt(2)*p['sigma'])))

    def Nsat_LRG(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        Mcut, M1 = 10**p['logMcut']*u.Msun, 10**p['logM1']*u.Msun
        return np.where(self.Mh>=p['kappa']*Mcut, ((self.Mh-p['kappa']*Mcut)/M1), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)
    
    """Thus, for satellite QSOs, we have [Eq 6]"""
    def Nsat_QSO(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMcut = 10**p['logMcut']*u.Msun
        M1 = 10**p['logM1']*u.Msun
        return np.where(self.Mh>=p['kappa']*logMcut, ((self.Mh-p['kappa']*logMcut)/M1), 0)**p['alpha']
    
    def Nsat(self, pdict={}, **kwargs):
        return self.Nsat_LRG(pdict, **kwargs) if "LRG" in self.Tracer else self.Nsat_QSO(pdict, **kwargs)
    
    # NOTE: placeholder, needs normalization
    def ucen(self, pdict={}, **kwargs):
        return self.dirac
    
    # NOTE: placeholder
    def usat(self, pdict={}, **kwargs):
        return self.NFW_norm
        
    
    


class Table3_4(ParamTable):
    # Table 3. LRG and QSO marginalized posteriors, with different models and different measurements. The error bars are 1𝜎 uncertainties. We also display several derived parameters, specifically the marginalized satellite fraction 𝑓sat, the sample completeness 𝑓ic, the average halo mass per galaxy log 𝑀h, and the linear bias𝑏lin. Units of mass are given in ℎ−1 𝑀⊙ .
    # Table 4. The results for the fits to high-z LRG sample with two redshift bin:0.8 < 𝑧 < 0.95 and 0.95 < 𝑧 < 1.1. We show the mean±1𝜎 error for HOD and derived parameters. We also list the average comoving number density in units of 10−4 (ℎ−1Mpc) −3. Masses are in units of ℎ−1 𝑀⊙ .
    def __init__(self, filename=f"{thispath}/Table3_4.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)

    
# Figure 1. The DESI One-Percent Survey LRG and QSO mean number density as a function of redshift. The dashed vertical lines show the fiducial LRG redshift bin edges of 𝑧 = 0.6, 𝑧 = 0.8, and the maximum redshift we consider for the QSO sample 𝑧 = 2.1.
class Fig1(BasePlots2):
    subplots = [[
        dict(name='Fig1', filename='Fig1', figsize=(6, 3.5),
             xlabel=r'$z$', xlim=(0.4, 2.3), xscale='linear',
             ylabel=r'$n(z) \ [h^3 \text{Mpc}^{-3}]$', ylim=(3e-6, 2e-3), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 7. The LRG HOD best-fit the posterior. The shaded regions correspond to 1 and 2𝜎 posteriors (68% and 95% intervals centered around the median prediction). The horizontal dotted line denotes 𝑁gal = 1.
class Fig7(BasePlots2):
    subplots = [[
        dict(name='Fig7a', filename='Fig7a', figsize=(7, 6),
             xlabel=r'$M_h [h^{-1} M_\odot]$', xlim=(1e12, 1e15), xscale='log',
             ylabel=r'$N_\text{gal}$', ylim=(1e-2, 1e1), yscale='log'),
        dict(name='Fig7b', filename='Fig7b', figsize=(7, 6),
             xlabel=r'$M_h [h^{-1} M_\odot]$', xlim=(1e12, 1e15), xscale='log',
             ylabel=r'$N_\text{gal}$', ylim=(1e-2, 1e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 10. The HOD posterior band (central+satellite) of LRG sample at𝑧 > 0.8. The results from 0.8 < 𝑧 < 0.95 and 0.95 < 𝑧 < 1.1 are shown in red and blue respectively. The shaded regions correspond to 1 and 2 𝜎posteriors.
class Fig10(BasePlots2):
    subplots = [[
        dict(name='Fig10', filename='Fig10', figsize=(6, 5),
             xlabel=r'$M_h \ [h^{-1} M_\odot]$', xlim=(1e12, 1e15), xscale='log',
             ylabel=r'$N^{(c+s)}_\text{gal}$', ylim=(3e-2, 1e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 14. The HOD posterior for the QSO sample. The shaded regions correspond to 1 and 2𝜎 posteriors.
class Fig14(BasePlots2):
    subplots = [[
        dict(name='Fig14', filename='Fig14', figsize=(6, 5),
             xlabel=r'$M_h \ [h^{-1} M_\odot]$', xlim=(1e12, 1e15), xscale='log',
             ylabel=r'$N_\text{gal}$', ylim=(3e-2, 1e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # satellite fraction, halo mass, bias, z-range, per zbin
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)



class HODParamsTable(ParamTable):  # best-fit HOD parameters, Tables 3 & 4
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        super().__init__(filename)





"""Old implementation being phased out"""



from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2024MNRAS.530..947Y
    subs = {'zbin': ['LRG1', 'LRG2', 'QSO', 'LRG3', 'LRG4']}
    info = {
        "f_sat": {"LRG1": 0.089, "LRG2": 0.104, "QSO": 0.05, "LRG3": 0.110, "LRG4": 0.151},
        "logMhMean": {"LRG1": 13.42, "LRG2": 13.26, "QSO": 12.74, "LRG3": 13.29, "LRG4": 13.00},  # Msun/h
        "b_lin": {"LRG1": 1.94, "LRG2": 2.11, "QSO": 2.56, "LRG3": 2.31, "LRG4": 2.13},
        
        # fixed cosmo params, 1.p7
        'Oc0h2': 0.1200, 'Ob0h2': 0.02237, 'sigma8': 0.811355, 'ns': 0.9649, 'h': 0.6736, 'w0':-1, 'wa':0,
        'mdef': '200c',  # M not clear, maybe same as zheng 2005/2007? or cmass?
        'MhMin': 1.3e11,  # Msun/h
        'zMin': {'LRG1': 0.6, 'LRG2': 0.8, 'QSO': 2.1, 'LRG3': 0.95, 'LRG4': 0.8},
        'zMax': {'LRG1': 0.4, 'LRG2': 0.6, 'QSO': 0.8, 'LRG3': 0.8, 'LRG4': 0.95},
    }
    info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    info['logMhMean'] = cycle(info['logMhMean'], lambda p, h=info['h']: np.log10(10**p/h))