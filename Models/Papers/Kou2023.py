"""
Cosmic census: Relative distributions of dark matter, galaxies, and diffuse gas

ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
arxiv.org/pdf/2211.07502
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Kou2023")

from scipy.special import erf



# 5.1 We recall that we fixed the cosmological parameters, for which we take the values from Planck Collaboration et al. (2020a) (TT, TE, EE+lowE+lensing+BAO): H0 =67.66 km s−1 Mpc−1, Ωbh2 = 0.2242, Ωch2 = 0.11933, τ =0.0561, ns = 0.9665, and σ8 = 0.8102.
class Cosmology():
    H0 = 67.66
    Ob0h2 = 0.02242
    Oc0h2 = 0.11933
    tau = 0.0561
    ns = 0.9665
    sigma8 = 0.8102


class HaloModel():
    # 4.2.3 We used the halo mass function that Tinker et al. (2008) defined and calibrated on numerical simulations,
    MassFunc = 'Tinker08'
    # 4.2.3 We made use of the simulation-fitted halo bias provided by Tinker et al. (2010)
    HaloBias = 'Tinker10'

    # 4.2 If not specified otherwise, we simply note M200m = M.
    MassDef = '200m'
    
    # 4.2 We worked with M500c for the gas profile and converted between M200m and M500c. In order to make this conversion, we followed Hu & Kravtsov (2003) and assumed a Navarro-FrenkWhite (NFW) profile (Navarro et al. 1997) and a concentration following Dolag et al. (2004), to remain consistent with the halo mass function derived by Tinker et al. (2008), who made use of this concentration-mass relation. 
    Concentration = 'Dolag04'
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        self.c = self.concentration()

    # 4.2.4 In the same way as for the mass conversion, we used the concentration parameter given by Dolag et al. (2004)... The concentration we used was then Eq 47 where c0 = 9.59 and αc = −0.102.
    def concentration(self):  # Eq 47
        c0, alpha_c = 9.59, -0.102
        return c0/(1+self.z) * (10**self.logM/(10**14))**alpha_c
    
    def radius200m(self):
        return (3 * 10**self.logM / (4 * np.pi * 200 * self.rhocrit))**(1/3)
    
    
class Data():
    # 2.1.1 We restricted our work to galaxies in the redshift range 0.47 <z < 0.59
    zMin, zMax = 0.47, 0.59
    
    # Abstract. Fitting a halo-based model to our measured angular power spectra (galaxy-galaxy, galaxy-lensing convergence, and galaxy-tSZ) at a median redshift of z = 0.53
    zMed = 0.53
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


class HOD_new():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = Table1().getparams(logMstarbin=self.logMstarbin).to_dict()
        except: self.p0 = {}
        
    # Section 4.2.1 We adopted an HOD parameterization following Zheng et al. (2005). The mean numbers of central and satellite galaxies in a halo of mass M are
    def Ncen(self, pdict={}, **kwargs): # Eq 19
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logM-p['logMmin'])/p['sigmalogM'])) * self.finc(pdict, **kwargs)

    # In order to reduce degeneracies, we restricted the number of free parameters by setting M0 = Mmin. We also fixed αg = 1, which is found by a number of studies (e.g., Zheng et al. 2005; Zehavi et al. 2011; More et al. 2015).
    def Nsat(self, pdict={}, **kwargs):  # Eq 20
        p = self.p0 | pdict | kwargs
        p['logM0'], p['alpha_g'] = p['logMmin'], 1
        return np.where(self.logM>=p['logM0'], ((10**self.logM-10**p['logM0'])/10**p['logM1']), 0)**p['alpha_g'] * self.Ncen(pdict, **kwargs)
    
    # mean is taken over all halos of mass M. The finc function was introduced by More et al. (2015) and enables us to take into account the fact that the CMASS sample is incomplete at low stellar masses. Leauthaud et al. (2016) showed that CMASS is about 80% complete at redshift 0.55 at stellar mass log(M/M⊙) = 11.4, while the completeness is very low at a low stellar mass. The finc function is defined as so that it is a function of the halo mass M and takes values between 0 and 1. This function is defined such that the completeness is 1 for masses M > Minc. The completeness therefore increases when Minc decreases or when αinc decreases.
    def finc(self, pdict={}, **kwargs):  # Eq 22
        p = self.p0 | pdict | kwargs
        return np.clip((1+p['alphainc']*(self.logM-p['logMinc'])), 0, 1)
    
    def ncen(self, pdict={}, **kwargs):
        return 1
    
    def nsat(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        rs = self.r200m/self.c
        p['gamma'], p['alpha'] = 1, 1
        NFW = 1/ ( (self.r/rs)**p['gamma'] * (1+(self.r/rs)**p['alpha'])**((p['beta_s']-p['gamma'])/p['alpha']))
        pass


class PowerSpectra():
    pass
    
        
# Table 1: Medians of the parameter posterior distributions and 68% credible intervals. Because we found that βm could not be constrained, we fixed it to 3 (hence recovering the NFW profile) and then fitted the other parameters. In this table, we report the value of βm corresponding to the maximum likelihood with all other parameters fixed to their best-fit values and that we denoteβm, best fit. The value of χ2/d.o. f is also given for the best fit.
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        self.dfraw = read_wide_table(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(self.dfraw)
    
# Fig. 4: Observed and best-fit galaxy angular auto-power spectra,Cggℓ , for the four different stellar mass threshold bins. The same colors are used for the observed error bars and the corresponding theoretical best fit in each stellar mass bin.
class Fig4(BasePlots2):
    subplots = [[
        dict(name='Fig4', filename='Fig4', figsize=(9, 6),
             xlabel=r'$\ell$', xlim=(3.9e1, 4.1e3), xscale='log',
             ylabel=r'$\ell(\ell+1)C_\ell^{gg}/(2\pi)$', ylim=(2e-2, 1.1e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Fig. 6: Observed and best-fit angular tSZ-galaxy cross-power spectra, Cygℓ , for the four different stellar mass threshold bins.
class Fig6(BasePlots2):
    subplots = [
        [dict(name='Fig6a', filename='Fig6a', figsize=(6, 4),
              xlabel=r'$\ell$', xlim=(4.1e1, 1.3e3), xscale='log',
              ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$', ylim=(2.5e-9, 2.5e-7), yscale='log'),
         dict(name='Fig6b', filename='Fig6b', figsize=(6, 4),
              xlabel=r'$\ell$', xlim=(4.1e1, 1.3e3), xscale='log',
              ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$', ylim=(2.5e-9, 2.5e-7), yscale='log')],
        [dict(name='Fig6c', filename='Fig6c', figsize=(6, 4),
              xlabel=r'$\ell$', xlim=(4.1e1, 1.3e3), xscale='log',
              ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$', ylim=(2.5e-9, 2.5e-7), yscale='log'),
         dict(name='Fig6d', filename='Fig6d', figsize=(6, 4),
              xlabel=r'$\ell$', xlim=(4.1e1, 1.3e3), xscale='log',
              ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$', ylim=(2.5e-9, 2.5e-7), yscale='log')],
    ]

    def __init__(self):
        super().__init__(thispath)


# Fig. 8: Galaxy distribution as a function of the host halo mass.Top: Total (central + satellite) number of galaxies contained in halos as a function of halo mass, M, evaluated at the sample median redshift. The four curves correspond to the different stellar mass threshold bins. The shaded bands represent the 1σ uncertainty. Middle: Total number of galaxies per comoving volume and halo mass range (in units of Mpc−3 M−1⊙ ) as a function of halo mass. Bottom: Total number of galaxies per halo mass range normalized by the number of galaxies in each stellar mass bin. This fraction of galaxies lies in a halo of mass M. The quantity Vappearing in the label stands for the comoving volume.
class Fig8a(BasePlots2):
    subplots = [[
        dict(name='Fig8a', filename='Fig8a', figsize=(9, 6),
             xlabel=r'$M \ [M_\odot]$', xlim=(1e12, 2e15), xscale='log',
             ylabel=r'$\langle N | M \rangle$', ylim=(5e-4, 3e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # per-mass-bin best-fit values referenced in Studies.info
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)




class HODParamsTable(ParamTable):  # best fit HOD parameters
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        self.df = read_wide_table(filename)




"""Old implementation I'm phasing out"""


from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
    subs = {'mbin':['M1', "M2", "M3", "M4"],}
    info = {
        # best fit HOD parameters
        "A": {"M1": 0.981, "M2": 0.965, "M3": 0.956, "M4": 0.961},  # cross-correlation amplitude
        "beta_m": {"M1": 4.97, "M2": 5.91, "M3": 4.16, "M4": 10},  # matter density profile
        
        # fixed cosmological parameters
        'h':0.6766, 'Ob0h2':0.02242, 'Oc0h2':0.11933, 'tau':0.0561, 'ns':0.9665, 'sigma8':0.8102, # 5.1p3
        'mdef': '200m',  # region in which the average density is ∆ = 200 times the cosmic mean density
        'MassFunc': 'Tinker08', 'Concentration':'Dolag04',
        'zlims': [0.47, 0.59],  # redshift range of selected galaxies
        'zmed': 0.53,  # median redshift
        'logMsMin': {'M1': 10.8, 'M2': 11.1, 'M3': 11.25, 'M4': 11.4},  # minimum stellar mass of selected
        'c0': 9.59, 'alpha_c': -0.102,  # concentration parameters, Eq47
    }
    
    def conc(self, z, logM):  # Eq 47
        return self.c0/(1+z) * (10**logM/(10**14))**self.alpha_c
    
from Models.HODs import BaseHOD
from Models.Papers import Zheng2005, More2015
from Models import HaloModels
class HOD(BaseHOD, Study):  # Kou 2023, arxiv.org/abs/2211.07502
    models = {'mbin':['M1', "M2", "M3", "M4"],
}
    params = {
        # best fit HOD parameters
        "logM_min": {"M1": 13.47, "M2": 13.58, "M3": 13.84, "M4": 14.20},  # minimum halo mass for a central galaxy/halos contain 0.5 central galaxies on average
        "sigma_logM": {"M1": 0.76, "M2": 0.78, "M3": 0.86, "M4": 0.959},  # changes the number of galaxies in low-mass halos
        "logM_1": {"M1": 14.119, "M2": 14.140, "M3": 14.171, "M4": 14.100},  # controls the number of galaxies at high halo mass
        "beta_s": {"M1": 4.38, "M2": 4.71, "M3": 5.31, "M4": 6.35},  # satellite galaxy profile
        "alpha_inc": {"M1": 0.51, "M2": 0.42, "M3": 0.39, "M4": 0.33},  # included to account for galaxy incompleteness at the low stellar mass end
        "logM_inc": {"M1": 13.39, "M2": 13.42, "M3": 13.69, "M4": 13.96},  # included to account for galaxy incompleteness at the low stellar mass end
    }
    # uses m200m
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 19, 21
        finc = lambda p: More2015.HOD().finc(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        func = lambda p: Zheng2005.HOD().Nc(logM, logMmin=p['logM_min'], sigmalogM=p['sigma_logM'])*finc(p)
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 20
        func = lambda p: Zheng2005.HOD().Ns(10**logM, M0=10**p['logM_min'], M1=10**p['logM_1'], alpha=1) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)

    def ns_r(self, rs, zs, logMs): #
        self.require(['r200m'])  # needs radius definition
        GNFW_func = lambda p: self.GNFW_r(rs, zs, logMs, self.r200m, self.conc)(beta=p['beta_s'])
        func = lambda p: GNFW_func(p)/np.trapezoid(GNFW_func(p)*4*np.pi*rs**2, rs, axis=0)
        return lambda p={}: func(self.p0 | p)

    def ns(self, ks, zs, logMs):  # Default satellite distribution (FFT of NFW)
        self.require(['c200m', 'r200m'])
        fft = HaloModels.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)
        return lambda p={}: fft.FFT3D(NFW(p))



        # rs, zs, logMs = self.setdim(rs, zs, logMs)
        # xs = rs*u.Mpc/(self.r200m(zs, logMs)/self.c200m(zs, logMs))  # scaled radius
        # NFW = 1 / (xs * (1+xs)**2)
        # NFW_int = np.trapezoid(NFW, xs, axis=0)
        # print(np.trapezoid(NFW, xs, axis=0), np.trapezoid(NFW, rs, axis=0))
        # return lambda p={}: NFW/NFW_int
        
class Measurement(Study):
    path = f"{DATA_PATH}/Kou2023"  # path to data, taken from plots using webplotdigitizer
    subs = {'mbin':['M1', "M2", "M3", "M4"]}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['mbin'])
            
        self.get_meas()

    def get_meas(self):
        with h5py.File(f'{self.path}/Kou2023_wpd.h5', "r") as f:
            self.Cgg_ell = f['dgg_ells'][()]
            self.Cgy_ell = f['dgy_ells'][()]
            self.Cgg_data = f[f'dgg_{self.mbin}'][()]/(self.Cgg_ell*(self.Cgg_ell+1))*2*np.pi
            self.Cgy_data = f[f'dgy_{self.mbin}'][()]/(self.Cgy_ell*(self.Cgy_ell+1))*2*np.pi
            
from Models.Profiles import BaseProfile
class HaloProfile(BaseProfile, Study):  # Planck 2018 and SDSS BOSS CMASS DR12
    models = {'mbin':['M1', "M2", "M3", "M4"],}  # mass bin
    params = {
        # Best-fit parameters
        "bh_m1": {"M1": 0.602, "M2": 0.623, "M3": 0.558, "M4": 0.550},  # hydrostatic bias
        # Fixed parameters
        'alpha_p': 0.12,  # Eq32
        'P0':6.41, 'gamma':0.31, 'alpha':1.33, 'beta':4.13, 'c_Pe':1.81, # eq 48
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Pe_del(self, xs):
        xs, _, _ = self.setdim(xs, 1, 1)  # set proper dimensions [nr, nz, nM]
        gnfw = lambda x, c, alpha, beta, gamma, P0: P0/((x*c)**gamma * (1+(x*c)**alpha)**((beta-gamma)/alpha))
        return gnfw(xs, c=self.p0['c_Pe'], P0=self.p0['P0'], gamma=self.p0['gamma'], alpha=self.p0['alpha'], beta=self.p0['beta'])

    def Pdel(self, zs, logMs, units='cosmo'):
        self.require(['H'])
        prefac = (1.65*(self.h/0.7)**2 * u.eV/u.cm**3 * (self.H(zs)/(self.H0))**(8/3)).to(self.units('pres', units))
        infac = (10**logMs/(3e14*0.7/self.h))
        Pdel = lambda p: prefac * (infac*p['bh_m1'])**(2/3+self.p0['alpha_p'])
        return lambda p={}: Pdel(self.p0 | p)
    
    def Pe1h(self, rs, zs, logMs, units='cosmo'):
        self.require(['r200m'])
        Pdel = self.Pdel(zs, logMs, units)
        logMs_ = lambda p: np.log10(p['bh_m1'])+logMs
        Pe_del = lambda p: self.Pe_del(rs*u.Mpc/self.r200m(zs, logMs_(p)))
        return lambda p={}: Pdel(self.p0 | p)*Pe_del(self.p0 | p)