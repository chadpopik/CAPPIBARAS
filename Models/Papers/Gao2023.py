"""
The DESI One-Percent Survey: Constructing Galaxy-Halo Connections for ELGs and LRGs Using Auto and Cross Correlations

ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
arxiv.org/pdf/2306.06317
"""


from config import *
from scipy.interpolate import interp1d
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Gao2023")


class Cosmology():
    # 3.1 This simulation adopts the standard cosmology: Ωm = 0.268, ΩΛ = 0.732, h = 0.71, ns = 0.968 and σ8 = 0.83
    h = 0.71
    Om0 = 0.268
    Ol0 = 0.732
    ns = 0.968
    sigma8 = 0.83


class HaloModel():
# The mass of a host halo Mh is defined as its current Mvir that is the mass enclosed by a virialized spherical structure with an over-density ∆vir (z) (Gunn & Gott 1972; Bryan & Norman 1998)
    MassDef = 'vir'  # Current Virial Mass


class SHMR():
    # The mass of a host halo Mh is defined as its current Mvir that is the mass enclosed by a virialized spherical structure with an over-density ∆vir (z) (Gunn & Gott 1972; Bryan & Norman 1998).
    MassDef = 'vir'
    
    def __init__(self, model, logMh=None, logMs=None, cosmology=Cosmology(), **kwargs):
        self.__dict__.update(locals())
        
        try: self.p0 = Table3().getparams(model=self.model).to_dict()
        except: self.p0 = {}
        
        if self.logMh is not None:
            self.Mh = 10**self.logMh/self.h*u.Msun
            self.logMh = np.log10(self.Mh/u.Msun)
            
        elif self.logMs is not None:
            self.logMh = np.linspace(10, 16, 1000)  # Covers the range of halo masses for interpolation
            self.Mh = 10**self.logMh*u.Msun
            self.Ms = 10**self.logMs*u.Msun
            

    # 3.2 We adopt a double power-law function (Wang et al. 2006; Wang & Jing 2010; Yang et al. 2012; Moster et al. 2013) to parameterize the⟨M∗|Mh⟩: Eq 6, where M0 divides the SHMR into two parts with different slopes α and β, and k is a normalization constant.
    def logMstar(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        M0, k = 10**p['log10M0']*u.Msun, 10**p['log10k']
        Mstar = 2*k / ((self.Mh/M0)**(-p['alpha']) + (self.Mh/M0)**(-p['beta']))
        return np.log10(Mstar)
    
    def logMhalo(self, pdict={}, **kwargs):
        logMstar_interp = self.logMstar(pdict, **kwargs)
        return interp1d(logMstar_interp, self.logMh, bounds_error=False, fill_value='extrapolate', kind='linear')(self.logMs)



# TABLE 3. Best-fit parameters of the SHMRs for different Psat models. Note—The first two columns represent the constant Psatmodel, and the third column denotes the halo mass-dependentPsat(Mh) model. The check marks in the first four rows indicate which observational quantities are used in the fit. The best-fit model parameters as well as 1σ uncertainties are shown in the remaining rows.
class Table3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        

class ParamsTable(ParamTable):  # best-fit SHMR parameters, Table 3
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)


# Figure 7. Constraints of SHMR (left panel) and Psat (right panel) models. The best-fit results (the first two columns of Table 3) for the constant Psat model using either ELG auto or LRGxELG cross correlation are displayed as orange and brown curves, respectively. The blue solid lines represent the best-fit result (the last column of Table 3) for the halo mass-dependent Psatmodel using all the correlation functions. The shallow regions denote the 1σ scatter. The horizontal gray line indicates the lowest stellar mass limit that can be probed by the ELG subsample.
class Fig7a(BasePlots2):
    subplots = [[
        dict(name='Fig7a', filename='Fig7a', figsize=(8, 6),
             xlabel=r'$\log(M_h) \ [M_\odot /h]$', xlim=(10, 15), xscale='linear',
             ylabel=r'$\log(M_*) \ [M_\odot]$', ylim=(7.5, 12), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Figure 2. Evolution of the SMFs of the LRG and ELG samples in the One-Percent survey. The left and right panels correspond to LRG and ELG respectively. The data points with Poisson errors denote the observed SMFs in different redshift bins. In the measurements, each galaxy has been multiplied by the completeness weight as mentioned in Section 2.1.
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(9, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(7, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)






"""Old implementation I'm phasing out"""




class SHMR_old():  # DESI 1% (arxiv.org/abs/2306.06317)
    models = {'model':["Auto", "Cross", "Psat"],}
    params = {
        # best-fit SHMR parameters, Table 3
        'logM0': {'Auto': 11.56, 'Cross': 12.14, 'Psat': 12.07},  # divides the slopes
        'alpha': {'Auto': 0.43,  'Cross': 0.37,  'Psat': 0.37},  # slope
        'beta': {'Auto': 2.72,  'Cross': 2.27,  'Psat': 2.61},  # slope
        'logk': {'Auto': 10.11, 'Cross': 10.40, 'Psat': 10.36},  # normalization constant
        'sigma': {'Auto': 0.18,  'Cross': 0.21,  'Psat': 0.21},  # scatter, TODO: what to do with this?
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def SHMR(self, logMh): # Eq 6
        self.require(['model'])
        logMh = logMh - np.log10(self.h)  # Mh -> Mh/h
        func = lambda p: np.log10(2*10**p['logk'] / ((10**logMh/10**p['logM0'])**(-p['alpha']) + (10**logMh/10**p['logM0'])**(-p['beta'])))
        return lambda p={}: func(self.p0 | p)
    
    # Get halo mass from stellar mass using interpolation
    def HSMR(self, logMs):
        logMhs = np.linspace(10, 20, 1000)  # Should cover the range of reasonable halo masses
        func = lambda p: np.interp(logMs, self.SHMR(logMhs)(self.p0 | p), logMhs)
        return lambda p={}: func(self.p0 | p)
    

from scipy.interpolate import RegularGridInterpolator
class TargetData():  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    path = f"{DATA_PATH}/Gao2023"
    subs = {'sample':['LRG', 'ELG']}  # Galaxy Sample

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])

    def make_zMsdists(self, dz=None, zMin=None, zMax=None, dlogMs=None, logMsMin=None, logMsMax=None):
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z_df = (zbins[1:]+zbins[:-1])/2
        self.dz_df = self.z_df[1]-self.z_df[0]
        
        # Read the plot data from the files
        self.logMs_df = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.n_logMs_z_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]
        
        self.dndzdlogMs_df = self.n_logMs_z_h3 *self.h**3 /u.Mpc**3 /self.dz_df/u.dex
        
        hmod = HaloModels.astropy_model(**Study.info)
        Vcoms = (hmod.Vcom(self.z_df+self.dz_df/2)-hmod.Vcom(self.z_df-self.dz_df/2)) *(self.area/(4*np.pi*u.sr).to(u.deg**2))  # Calculate non-comoving shell for every z
        self.dNdzdlogMs_df = self.dndzdlogMs_df * Vcoms[:, None]
        
        dNinterp = RegularGridInterpolator((self.z_df, self.logMs_df), self.dNdzdlogMs_df,bounds_error=False, fill_value=0)

        zmin = zMin if zMin is not None else self.z_df.min()
        zmax = zMax if zMax is not None else self.z_df.max()
        self.dz = dz if dz is not None else self.z_df[1]-self.z_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)

        logMsMin = logMsMin if logMsMin is not None else self.logMs_df.min()
        logMsMax = logMsMax if logMsMax is not None else self.logMs_df.max()
        self.dlogMs = dlogMs if dlogMs is not None else self.logMs_df[1]-self.logMs_df[0]
        self.logMs = np.arange(logMsMin, logMsMax+self.dlogMs, self.dlogMs)

        zgrid, logMsgrid = np.meshgrid(self.z, self.logMs, indexing='ij')
        self.dNdzdlogMs = dNinterp(np.column_stack([zgrid.ravel(), logMsgrid.ravel()])).reshape(len(self.z), len(self.logMs)) / u.dex
        
        self.dNdogMs_z = self.dNdzdlogMs *self.dz
        self.N_z = np.trapezoid(self.dNdogMs_z, self.logMs)
        self.n_z = self.N_z / self.area
        self.dNdz = self.N_z / self.dz
        self.dndz = self.dNdz / self.area
        
        self.N_z_logMs = self.dNdogMs_z *self.dlogMs*u.dex
        

    # def dndlogmstar(self, hh=0.71, **kwargs):  # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    #     return self.dndlogmstar_h3*hh**3
    
    # def dNdz(self, **cosmopars):
    #     return np.trapezoid(self.dndlogmstar(**cosmopars), self.logmstar)*self.volumes(**cosmopars)/(self.z[1]-self.z[0])