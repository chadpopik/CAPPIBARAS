"""
Cosmic census: Relative distributions of dark matter, galaxies, and diffuse gas

ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
arxiv.org/pdf/2211.07502
"""


from config import *

from scipy.special import erf

from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


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


class HOD():
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

