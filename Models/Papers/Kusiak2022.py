"""
Constraining the galaxy-halo connection of infrared-selected u n W I S E galaxies with galaxy clustering and galaxy-CMB lensing power spectra

ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
arxiv.org/pdf/2203.12583
"""


from config import *

from scipy.special import erf

from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Kusiak2022")


class Cosmology():
    pass
    # 1. Throughout this analysis, we assume a flat ΛCDM cosmology with Planck 2018 best-fit parameter values (last column of Table II of Ref. [16]): ωcdm = 0.11933, ωb = 0.02242, H0 = 67.66 km/s/Mpc, ln(1010As) = 3.047 andns = 0.9665 with kpivot = 0.05 Mpc−1, and τreio = 0.0561.
    
    # 1. In our analysis, we work in units of M /h for masses and we adopt the M200chalo mass definition everywhere, i.e., the mass enclosed within the spherical region whose density is 200 times the critical density of the universe, and the corresponding mass-dependent radius r200c, which encloses mass M200c
    

    
class HaloModel():
    pass
    # B1. dn/(dM ) is the differential number of halos per unit mass and volume, defined by the halo mass function (HMF), where in our analysis we use the Tinker et al. analytical fitting fuction [39]
    
    # In class_sz, we set the mass bounds of the integral to Mmin = 7 × 108 M /hand Mmax = 3.5 × 1015 M /h and the redshift bounds to zmin = 0.005 and zmax = 4, the latter dictated by the upper redshift limit of the unWISE galaxy samples that we analyze.
    
    
class HOD_new():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: 
            if self.params=='best-fit':
                self.p0 = Table4().getparams(Sample=self.Sample).to_dict()
            elif self.params=='posterior':
                self.p0 = Table5().getparams(Sample=self.Sample).to_dict()
            self.p0['logM0'] = self.p0['logMHODmin']
        except: self.p0 = {}
                
    def Ncen(self, pdict={}, **kwargs): # Eq 4
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logM-p['logMHODmin'])/(p['sigmalogM'])))

    def Nsat_LRG(self, pdict={}, **kwargs):  # Eq 5
        p = self.p0 | pdict | kwargs
        return np.where(10**self.logM>=10**p['logM0'], ((10**self.logM-10**p['logM0'])/10**p['logMHOD1']), 0)**p['alphas'] * self.Ncen(pdict, **kwargs)
    
# TABLE IV: Best-fit values for the six model parameters obtained by jointly fitting the measured unWISE and Planck galaxygalaxy auto- and galaxy-CMB lensing cross-correlation to the halo model predictions, along with the χ2 and PTE for the best-fit (for 19 data points, i.e., 13 degrees of freedom), for each of the three unWISE galaxy samples. We also include results for five derived parameters: M ′1 and M HOD min in units of M /h, the fraction of satellite galaxies αsat (see Eq. 29), the mean galaxy bias bg (see Fig. 11), as well as the average host halo mass Mh (see also Fig. 12). The latter three are computed with the best-fit values of the HOD parameters from this table.
class Table4(ParamTable):
    def __init__(self, filename=f"{thispath}/Table4.csv"):
        self.df = read_wide_table(filename)
        
        
# TABLE V: Statistical summary of the posteriors (mean and 68% marginalized constraints) for the six model parameters obtained by jointly fitting the measured unWISE and Planck galaxy-galaxy auto- and galaxy-CMB lensing cross-correlations to the halo model predictions, separately for each of the three unWISE galaxy samples. The 1D and 2D marginalized posterior distributions are shown in Fig. 7. We also provide results for two derived parameters: M ′1 and M HOD min (in units of M /h), as well as the mean galaxy bias bg (see Fig. 11), the average host halo mass Mh (see also Fig. 12 , and the fraction of satellite galaxies αsat (Eq. 29). αsat, bg , and Mh and their error bars (also corresponding to the 68% CL) are obtained by computingαsat for the last 80,000 steps of the MCMC chains, which constitutes about half of the samples.
class Table5(ParamTable):
    def __init__(self, filename=f"{thispath}/Table5.csv"):
        self.dfraw = read_wide_table(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(self.dfraw)
    
    
# FIG. 2: Normalized redshift distributions 1N totgdNg /dz for each of the unWISE galaxy samples: blue (solid), green (dashed), and red (dotted), obtained by cross-matching the unWISE objects with the COSMOS catalog. Other important characteristics of the unWISE samples are presented in Table III.
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 4),
             xlabel=r'$z$', xlim=(0, 4), xscale='linear',
             ylabel=r'$\frac{1}{N_g^\text{tot}} \frac{dN_g}{dz}$', ylim=(-0.06, 1.3), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 8: Measurements of the galaxy-galaxy auto-power spectrum, Cgg` , and the CMB lensing-galaxy cross-power spectrum,Cgκcmb` , for each of the unWISE galaxy samples, along with our halo model theory curves for the best-fit model parameters (Table IV). The unWISE galaxy samples are color-coded from top to bottom: blue, green, and red, with Cgg ` on the left and Cgκcmb` on the right. On each galaxy auto-power spectrum plot, the solid curves are the best-fit total signal, the dotted curves show the best-fit 1-halo contribution to Cgg ` , the dashed show the best-fit 2-halo contribution to Cgg ` , the dash-dotted black show the total best-fit lensing magnification contribution, and the grey dash-dot-dotted show the best-fit shot noise contribution. On the CMB lensing-galaxy cross-spectra plots, the solid curves show the best-fit total signal, the dotted curves show the best-fit 1-halo contribution to Cgκcmb` , and the dashed show the best-fit 2-halo contribution to Cgκcmb` ; the lensing magnification contributions are 3-4 orders of magnitude smaller than the presented curves and therefore not shown in the CMB lensing case. Note that in the Cgg ` plots the y-axis is shown on a linear scale, while for the Cgκcmb` plots it is on a logarithmic scale. Each plot has a bottom panel that shows the residuals of the best-fit model for each bin.
class Fig8_col1(BasePlots2):
    subplots = [
        [dict(name='Fig8a', filename='Fig8a', figsize=(6, 3.33),
              xlabel=r'$\ell$', xlim=(1e2, 1e3), xscale='linear',
              ylabel=r'$10^5 \times C^{gg}_\ell$', ylim=(-0.01, 0.75), yscale='linear')],
        [dict(name='Fig8c', filename='Fig8c', figsize=(6, 3.33),
              xlabel=r'$\ell$', xlim=(1e2, 1e3), xscale='linear',
              ylabel=r'$10^5 \times C^{gg}_\ell$', ylim=(-0.01, 0.75), yscale='linear')],
        [dict(name='Fig8e', filename='Fig8e', figsize=(6, 3.33),
              xlabel=r'$\ell$', xlim=(1e2, 1e3), xscale='linear',
              ylabel=r'$10^5 \times C^{gg}_\ell$', ylim=(-0.01, 0.75), yscale='linear')],
    ]

    def __init__(self):
        super().__init__(thispath)


# FIG. 9: Mean number of central and satellite galaxies, Nc and Ns, versus halo mass for the unWISE samples, computed for the mean posterior values of the HOD parameters (Table V). The solid lines show Ns and the dashed lines show Nc. Top left: all three unWISE samples on one plot. Top right: blue sample. Bottom left: green sample. Bottom right: red sample. For the individual plots, we also include the prediction computed for the best-fit values of the HOD parameters (Table IV) in thinner lines. The light grey (dark grey) regions show the Ns (Nc) curves computed for the HOD parameter values from the last 80,000 steps of the MCMC chains to illustrate the uncertainty on the mean number of satellite (central) galaxies.
class Fig9(BasePlots2):
    subplots = [
        [dict(name='Fig9a', filename='Fig9a', figsize=(7.5, 6),
              xlabel=r'Mass [$M_\odot/h$]', xlim=(2e11, 5e15), xscale='log',
              ylabel=r'mean number of galaxies', ylim=(1e-2, 1e2), yscale='log'),
         dict(name='Fig9b', filename='Fig9b', figsize=(7.5, 6),
              xlabel=r'Mass [$M_\odot/h$]', xlim=(2e11, 5e15), xscale='log',
              ylabel=r'mean number of galaxies', ylim=(1e-2, 1e2), yscale='log')],
        [dict(name='Fig9c', filename='Fig9c', figsize=(7.5, 6),
              xlabel=r'Mass [$M_\odot/h$]', xlim=(2e11, 5e15), xscale='log',
              ylabel=r'mean number of galaxies', ylim=(1e-2, 1e2), yscale='log'),
         dict(name='Fig9d', filename='Fig9d', figsize=(7.5, 6),
              xlabel=r'Mass [$M_\odot/h$]', xlim=(2e11, 5e15), xscale='log',
              ylabel=r'mean number of galaxies', ylim=(1e-2, 1e2), yscale='log')],
    ]

    def __init__(self):
        super().__init__(thispath)


# FIG. 15: Impact of varying two different parametrizations λ ≡ rout/r200c (left) and a ≡ csat/cdm (right) on the Fourier transform of the truncated NFW profile (Eq. 8). Our analysis uses the parametrization in the left panel, while others (e.g., [10]) in the literature have used the parametrization in the right panel. For this plot, we use ∆ = 200 with M200c = 3 × 1014 M /h atz = 1. We also set χ = 1317 Mpc/h (which can be used to convert between ` and k = (` + 0.5)/χ). For this example halo,c200c = 3.4 computed with the concentration-mass relation from [46]. Our fiducial truncation radius is r200c.
class Fig15(BasePlots2):
    subplots = [[
        dict(name='Fig15a', filename='Fig15a', figsize=(6, 4),
             xlabel=r'$\ell$', xlim=(7e1, 1.5e5), xscale='log',
             ylabel=r'$u_\ell^\text{m}$', ylim=(-0.05, 1.05), yscale='linear'),
        dict(name='Fig15b', filename='Fig15b', figsize=(6, 4),
             xlabel=r'$\ell$', xlim=(7e1, 1.5e5), xscale='log',
             ylabel=r'$u_\ell^\text{m}$', ylim=(-0.05, 1.05), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # best fit HOD params + Table II, per sample
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)




class HODParamsTable(ParamTable):  # best fit HOD params
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        self.df = read_wide_table(filename)




"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
    subs = {'sample':['Blue', 'Green', 'Red'],}

    info = {
        # best fit HOD params
        "ASNe7": {"Blue": -0.16, "Green": 1.35, "Red": 27.95},
        # Table II
        'zMean': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5},
        'ndens': {'Blue': 3409, 'Green': 1846, 'Red': 144},
        'area': 0.586*4 * np.pi * (180/np.pi)**2*u.deg**2,
        
        # fixed cosmo params
        'Oc0h2': 0.11933, 'Ob0h2': 0.02242, 'h':0.6766, 'ns':0.9665, 'lnAsn10': 3.047, 'kpivot':0.05, 'tau_reio':0.0561,  # Ip7
        # HaloModel choices, Eq 10&30, Section IpLast
        'MassDef': '200c', 'Concentration': 'Bhattacharya13', 'MassFunc': 'Tinker08', 'HaloBias': 'Tinker10',
        # Other info
        'MhMin': 7e8, 'MhMax': 3.5e15,  # Msun/h
        'zMin_hmod': 0.005, 'zMax_hmod': 4,
        'zMin': 0, 'zMax': 2,
        'logM0': 0,
            }

    info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    info['MhMax'] = cycle(info['MhMax'], lambda M, h=info['h']: M*u.Msun/h)
    info['ndens'] = cycle(info['ndens'], lambda n: n/u.deg**2)
    
    
    
    
    
from Models.HODs import BaseHOD, Ci, Si
from Models.Papers import Zheng2005
class HOD(BaseHOD, Study):  # Kusiak 2022, arxiv.org/abs/2203.12583
    models = {'sample':['Blue', 'Green', 'Red'],}
    params = {
        # best fit HOD params
        "sigma_logM": {"Blue": 0.73, "Green": 0.61, "Red": 0.75},
        "alpha_s": {"Blue": 1.38, "Green": 1.23, "Red": 1.18},
        "logM_min_HOD": {"Blue": 12.11, "Green": 12.39, "Red": 13.23}, # Msun/h
        "logM_1": {"Blue": 13.00, "Green": 12.87, "Red": 13.20}, # Msun/h
        "Lambda": {"Blue": 1.11, "Green": 2.50, "Red": 1.30},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logMs):
        func = lambda p: Zheng2005.HOD().Nc(logMs-np.log10(self.h), logMmin=p['logM_min_HOD'], sigmalogM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logMs):
        func = lambda p: Zheng2005.HOD().Ns(10**logMs/self.h, M0=0, M1=10**p['logM_1'], alpha=p['alpha_s']) * self.Ncen(logMs)(p)
        return lambda p={}: func(self.p0 | p)

    def nsat(self, ks, zs, logMs):  # Eq 8-10
        self.require(['rhom', 'c200c', 'r200c'])
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        qs0 = ks/u.Mpc*self.r200c(zs, logMs)
        qs = qs0/self.c200c(zs, logMs)  # Define scaled wavenumber

        NFW_trunc = lambda L: (np.cos(qs*u.rad) * (Ci(qs+qs0*L)-Ci(qs)) + np.sin(qs*u.rad) * (Si(qs+qs0*L)-Si(qs)) - np.sin(qs0*L*u.rad)/(qs+qs0*L))

        f_NFW = lambda x: (np.log(1+x)-x/(1+x))**(-1)
        prefac = 10**logMs*u.Msun/self.rhom(0) # density from halo mass over mean density at z=0
        func = lambda p: NFW_trunc(p['Lambda'])*f_NFW(self.c200c(zs, logMs)*p['Lambda'])
        return lambda p={}: prefac*func(self.p0 | p)
    
    
    
    
class Kusiak2022(Study):  # HOD for unWISE galaxies and Planck lensing arxiv.org/abs/2203.12583
    path = f"{DATA_PATH}/Kusiak2022"  # path to data, taken from plots using webplotdigitizer
    subs = {'sample':['Blue', 'Green', 'Red']}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])
        
        self.get_meas()
        _, _ = self.get_dNdz(20)
        
        
        
        
from Models.TargetData import BaseTargetData
class TargetData(BaseTargetData, Study):  # unWISE galaxies and Planck lensing
    path = f"{DATA_PATH}/Kusiak2022"  # path to data, provided by author
    subs = {'sample':['Blue', 'Green', 'Red']}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def make_zdists(self, dz=None, zMin=None, zMax=None):
        self.require(['sample'])
        # load from digitized data
        sampstr = {'Blue':0,'Green':1,'Red':2}[self.sample]
        self.zs_df, dNdz_norm = np.loadtxt(f"{self.path}/normalised_dndz_cosmos_{sampstr}.txt").T  # normalized differential number count
        
        zmin = zMin if zMin is not None else self.zs_df.min()
        zmax = zMax if zMax is not None else self.zs_df.max()
        self.dz = dz if dz is not None else self.zs_df[1]-self.zs_df[0]
        self.z = np.arange(zmin, 
                           zmax+self.dz, self.dz)
        self.dNdz_norm = np.interp(self.z, self.zs_df, dNdz_norm)  # interpolate to desired z
        
        self.dNdz = self.dNdz_norm*self.area*self.ndens
        self.dndz = self.dNdz_norm*self.ndens
        self.N_z = self.dNdz_norm*self.area*self.ndens*self.dz
        self.n_z = self.dNdz_norm*self.ndens*self.dz
        
        
        
from Models.Spectra import BaseSpectra
class Spectra(BaseSpectra, HOD, TargetData):  # unWISE galaxies and Planck lensing (Kusiak+ 2023, arxiv.org/abs/2203.12583)

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)        
        logMs = np.linspace(7e8, 3.5e15, 20)/Study().h
        zs = np.linspace(0.005, 4, 10)
        
    def d2VdzdOmega(self, zs):
        self.require(['H', 'chi'])
        return (c.c*self.chi(zs)**2/self.H(zs)).decompose()
    
    def ugl(self, ks, zs, logMs):
        self.require(['ells'])
        nsat, Pkell = self.ngal(zs, logMs), self.k_to_ell(self.ells[:, None, None], ks, zs)
        return lambda p: Pkell(nsat(p))
    
    def C_ij(self):  # Eq. 3
        return lambda C1h_ij, C2h_ij: C1h_ij+C2h_ij
    
    def C1h_ij(self, zs, logMs, **kwargs):  # Eq. 4
        self.require(['dndlogM'])
        intfactor = self.dndlogM(zs, logMs)*self.d2VdzdOmega(zs)
        return lambda u_i, u_j: np.trapezoid(np.trapezoid(intfactor*u_i*u_j, logMs), zs)

    def C2h_ij(self, ks, zs, logMsi, logMsj, **kwargs):  # Eq. 5
        self.require(['Plin', 'dndlogM', 'bh', 'ells'])
        Plin_k = self.k_to_ell(self.ells, ks[:, :, 0], zs[:, 0])(self.Plin(ks, zs)[:, :, 0])[:, :, None]
        intfac_i, intfac_j = Plin_k*self.d2VdzdOmega(zs)*self.dndlogM(zs, logMsi)*self.bh(zs, logMsi), self.dndlogM(zs, logMsj)*self.bh(zs, logMsj)
        return lambda u_i, u_j: np.trapezoid(np.trapezoid(intfac_i*u_i, logMsi)*np.trapezoid(intfac_j*u_j, logMsj), zs)

    def u_g(self, ks, zs, logMs, **kwargs):  # Eq. 11
        Wg, Nc, Ns, ugl, ngal = self.W_g(zs), self.Nc(logMs), self.Nc(logMs), self.ugl(ks, zs, logMs), self.ngal(zs, logMs)
        return lambda p: Wg/ngal(p) * (Nc(p)+Ns(p)*ugl)
    
    def ngal(self, zs, logMs, **kwargs):  # Eq. 12
        Nc, Ns, dndlogm = self.Nc(logMs), self.Nc(logMs), self.dndlogM(zs, logMs)
        return lambda p: np.trapezoid((Nc+Ns)*dndlogm, logMs)
    
    def W_g(self, zs):  # Eq 13 & 14
        self.require(['dNdz', 'H', 'chi'])
        phig = self.dNdz/np.trapezoid(self.dNdz, zs)
        return (self.H(zs)/c.c*phig/self.chi(zs)).decompose()

    def C1h_gg(self, ks, zs, logMs):  # Eq. 15
        C1hij, ug2 = self.C1h_ij(zs, logMs), self.u2_g(ks, zs, logMs)
        return lambda p: C1hij(ug2(p), 1)

    def u2_g(self, ks, zs, logMs):  # Eq. 16
        Wg, Ns, ugl, ngal = self.W_g(zs), self.Ns(logMs), self.ugl(ks, zs, logMs), self.ngal(zs, logMs)
        return lambda p: Wg**2/ngal(p)**2 * (Ns(p)**2*ugl(p)**2 + 2*Ns*ugl(p))

    def C2h_gg(self, ks, zs, logMs):  # Eq. 17
        C2hij, ug = self.C2h_ij(ks, zs, logMs, logMs), self.u_g(ks, zs, logMs)
        C2hgg = lambda ug: C2hij(ug)
        return lambda p: C2hgg(ug(p))
    
    def C_gg(self, ks, zs, logMs):
        C1hgg, C2hgg = self.C1h_gg(ks, zs, logMs), self.C2h_gg(ks, zs, logMs)
        return lambda p: C1hgg(p) + C2hgg(p)

    # def u_g(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g / ngal(p) * (Nc(p)+Ns(p)*k_to_ell(usk(p)))
    
    # def ngal(self, Nc, Ns, hmf, logM, zs, **kwargs):
    #     hmf
    #     return lambda p={}: np.trapezoid((Nc(p)+Ns(p))*hmf, logM)
        
    # def W_g(self, Hz, chis, dNdz, zs, **kwargs):
    #     phi_g = dNdz / np.trapezoid(dNdz, zs)
    #     return (Hz/c.c.to(u.km/u.s).value * phi_g/chis**2)[:, None]
    
    # def C1h_gg(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ells, ks, **kwargs):
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     ug2 = self.ug2(Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs)
    #     return lambda p={}: np.trapezoid(d2V_dzdOmega*np.trapezoid(hmf*ug2(p), logM), zs)

    # def ug2(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     ul = lambda p: k_to_ell(usk(p))
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g**2/ngal(p)**2 * (Ns(p)**2*ul(p)**2 + 2*Ns(p)*ul(p))

    # def C2h_gg(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, Plin, bh, **kwargs):
    #     Plinl = self.uk_to_ul(ells, ks, chis, zs)(Plin)
    #     ug = self.u_g(ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks)
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     return lambda p={}: np.trapezoid(d2V_dzdOmega*Plinl*np.trapezoid(bh*hmf*ug(p), logM)**2, zs)

    # def SN(self, area, dNdz, ells, zs, **kwargs):
    #     return area*(u.deg**2).to(u.sr)/np.trapezoid(dNdz, zs) *np.ones(ells.shape)