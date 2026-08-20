"""
HOD models
"""

import numpy as np
import scipy
from scipy.special import erf, sici
import Models.Studies as Studies
import Models.HaloModels as HaloModels
import astropy.units as u


def Si(q):
    return scipy.special.sici(q.value)[0]

def Ci(q):
    return scipy.special.sici(q.value)[1]

class BaseHOD:
    def setdim(self, rs_ks, zs, logMs):  # Set proper dimensions of rs/ks, zs, Ms
        rs_ks, zs, logMs = (rs_ks.unit*np.array(rs_ks, ndmin=1))[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)
        return rs_ks, zs, logMs

    def ncen(self, ks, zs, logMs):  # Default central distribution [n density] (FFT of dirac delta=1)
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        return lambda *args, **kwargs: np.ones_like(ks*zs*logMs)/u.Mpc**2
    
    def nsat(self, ks, zs, logMs):  # Default satellite distribution [n density] (FFT of NFW profile)
        info = self.info.copy()
        if not hasattr(self, 'Concentration'): info['Concentration'] = 'Bhattacharya13'
        if not hasattr(self, 'MassDef'): info['MassDef'] = 'vir'
        hmod = HaloModels.pyccl_model(**info)
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        return lambda *args, **kwargs: hmod.NFW_k(ks, zs, logMs, trunc=True)/(10**logMs*u.Msun)
    
    def Ntot(self, logM, **kwargs):
        Nc, Ns = self.Ncen(logM), self.Nsat(logM)
        return lambda p={}: Nc(p)+Ns(p)
        
    
    # def ns(self, ks, zs, logMs):  # Default satellite distribution [n density] (FFT of NFW)
    #     ks, zs, logMs = self.setdim(ks, zs, logMs)
    #     rS = self.hmod.rdel(zs, logMs)/self.hmod.c(zs, logMs)
    #     qs = ks*rS
    #     rhok = 4*np.pi*rho*rS**3 * (np.cos(qs)*(Ci((1+c)*qs)-Ci(qs)) + np.sin(q)*(Si((1+c)*q)-Si(q)) - np.sin(c*q)/(1+c*q))
    #     rho0 = delta/3 * cdel**3/(np.log(1+cdel)-cdel/(1+cdel)) * rhodel
    #     nfwprof = rho0/(r/rs*(1+(r/rs))**2)

    # def A_NFW(self, c):
    #     return (np.log(1+c)-c/(1+c))**(-1)
    #     # return (1+c)/(np.log(1+c)*(1+c)-c) # equivalent

    # def NFW_q(self, qs, cs):  # Default satellite distribution (FFT of NFW profile)
    #     qs0 = ks/u.Mpc*(rdel(zs, logMs)/cdel(zs, logMs))  # define scaled wavenumber
    #     qs = lambda f_tr: qs0/f_tr  # apply truncation
    #     cs0 = cdel(zs, logMs)  # concentration
    #     cs = lambda f_tr: cs0*f_tr  # apply truncation
    #     Si, Ci = lambda q: sici(q)[0], lambda q: sici(q)[1]  # Define sine/cosine integrals
    #     nfw = lambda c, q: (np.cos(q)*(Ci((1+c)*q)-Ci(qs)) + np.sin(q)*(Si((1+c)*q)-Si(q)) - np.sin(c*q)/(1+c*q))  # define NFW
    #     return 

    # def ns_r(self, rs, zs, logMs):
    #     fft = FFTs.mcfit_package(rs=rs)
    #     NFW = self.ns(fft.ks, zs, logMs)
    #     func = lambda p: fft.IFFT3D(NFW(p))
    #     return lambda p={}: func(self.p0 | p)

    

    



class Zehavi2005(BaseHOD, Studies.Zehavi2005):  # virial
    models = {}
    params = {}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Nc(self, logM, logMmin):
        return np.heaviside(logM - logMmin, 1)

    def Ns(self, M, M1, alpha):
        return (M/M1)**alpha



"""In this paper, we assume that the satellite galaxy distribution
follows the dark matter distribution within the halo, which we describe by a spherically symmetric
NFW profile (Navarro, Frenk, & White 1995, 1996, 1997) truncated at the virial radius (defined to
enclose a mean overdensity of 200). The satellite-satellite galaxy pair distribution F ′ss(x) is then
the convolution of the NFW profile with itself (see Sheth et al. 2001). For the dependence of NFW
halo concentration on halo mass, we use the relation given by Bullock et al. (2001), after modifying
it to be consistent with our slightly different definition of the halo virial radius
"""


class Zheng2005(BaseHOD, Studies.Zheng2005):  # virial mass
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
        if self.nparams=='3P': func = lambda p: Zehavi2005().Nc(logM, logMmin=p['logMmin'])
        elif self.nparams=='5P': func = lambda p: self.Nc(logM, logMmin=p['logMmin'], sigmalogM=p['sigmalogM'])
        return lambda p={}: func(self.p0 | p)
    
    def Nsat(self, logM):
        if self.nparams=='3P': func = lambda p: Zehavi2005().Ns(10**logM, M1=10**p['logM1'], alpha=p['alpha'])
        elif self.nparams=='5P': func = lambda p: self.Ns(10**logM, M0=10**p['logM0'], M1=10**p['logM1'], alpha=p['alpha'])
        return lambda p={}: func(self.p0 | p)



class More2015(BaseHOD, Studies.More2015):  # BOSS DR11, arxiv.org/abs/1407.1856
    models = {'mbin': ['MA', 'MB', 'MC']}
    params = {
        # Best-fit parameters, Table 1
        "logMmin": {"MA": 13.13, "MB": 13.45, "MC": 13.68}, # I think Msol/h^2
        "sigma^2": {"MA": 0.22, "MB": 0.45, "MC": 0.79},
        "logM1": {"MA": 14.21, "MB": 14.51, "MC": 14.56}, # TODO: check units?
        "alpha": {"MA": 1.13, "MB": 1.14, "MC": 1.00},
        "kappa": {"MA": 1.25, "MB": 0.85, "MC": 1.19},
        "alpha_inc": {"MA": 0.44, "MB": 0.53, "MC": 0.57},  # incompleteness nuisance parameter
        "logM_inc": {"MA": 13.57, "MB": 13.88, "MC": 14.08},  # 
        "p_off": {"MA": 0.34, "MB": 0.37, "MC": 0.36},  # miscentering nuisance parameter
        "R_off": {"MA": 2.2, "MB": 2.3, "MC": 2.4},  # miscentering nuisance parameter
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
    def finc(self, logM, alpha_inc, logM_inc, written=True):  # Eq 5
        if written:
            return np.clip((1+alpha_inc*(logM-logM_inc)), 0, 1)
        else:
            return 2*np.clip((1+alpha_inc*(logM-logM_inc))/2, 0, 1)

    def Ncen(self, logM):  # Eq 3
        func = lambda p: self.finc(logM=logM-np.log10(self.h**2), alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'], written=False)/2 * Zheng2005().Nc(logM-np.log10(self.h**0), logMmin=p['logMmin'], sigmalogM=p['sigma^2']**0.5)
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 4
        func = lambda p: Zheng2005().Ns(M=10**logM/self.h**2, M0=p['kappa']*10**p['logMmin'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)

    def nc(self, k, z, logM):  # Eq 9 (transform of Eq 11)
        self.require('r200m')
        rs = self.r200m(z, logM)  # 
        func = lambda p: 1-p['p_off']+p['p_off'] * np.exp(-1/2 * k**2 * (rs*p['R_off'])**2)
        return lambda p={}: func(self.p0 | p)

    def ns(self, k, z, logM):  # TODO: add
        pass
    


class Kusiak2022(BaseHOD, Studies.Kusiak2022):  # Kusiak 2022, arxiv.org/abs/2203.12583
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
        func = lambda p: Zheng2005().Nc(logMs-np.log10(self.h), logMmin=p['logM_min_HOD'], sigmalogM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logMs):
        func = lambda p: Zheng2005().Ns(10**logMs/self.h, M0=0, M1=10**p['logM_1'], alpha=p['alpha_s']) * self.Ncen(logMs)(p)
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




class Yuan2023(BaseHOD, Studies.Yuan2023):  # DESI 1% LRGs and QSO using ABACUSHOD
    models = {'model':['Base', 'Ext'],  # model, Zheng07+fic or with added velocity bias
            'sample': ['LRG1', 'LRG2', 'QSO', 'LRG3', 'LRG4'],  # sample of galaxies
            }
    params = {
        # best-fit HOD parameters, Tables 3 & 4
        "logM_cut": {'Base': {"LRG1": 12.89, "LRG2": 12.78, "QSO": 12.67, "LRG3": 12.89, "LRG4": 12.68},
                     'Ext': {"LRG1": 12.79, "LRG2": 12.64, "QSO": 12.2}},  # Msun/h
        "logM_1": {  # roughly sets the typical halo mass that hosts one satellite galaxy, Msun/h
            'Base': {"LRG1": 14.08, "LRG2": 13.94, "QSO": 15.00, "LRG3": 13.96, "LRG4": 13.60},
            'Ext': {"LRG1": 13.88, "LRG2": 13.71, "QSO": 14.7}},
        "sigma": {   # controls the steepness of the transition from 0 to 1 in the number of central galaxies
            'Base': {"LRG1": 0.27, "LRG2": 0.23, "QSO": 0.58, "LRG3": 0.37, "LRG4": 0.53},
            'Ext': {"LRG1": 0.21, "LRG2": 0.09, "QSO": 0.12}},
        "alpha": {  # power law index on the number of satellite galaxies
            'Base':{"LRG1": 1.20, "LRG2": 1.07, "QSO": 1.09, "LRG3": 0.91, "LRG4": 0.72},
            'Ext':{"LRG1": 1.07, "LRG2": 1.18, "QSO": 0.8}},
        "kappa": {  # xMcut gives the minimum halo mass to host a satellite galaxy
            'Base': {"LRG1": 0.65, "LRG2": 0.55, "QSO": 0.74, "LRG3": 0.74, "LRG4": 0.51},
            'Ext': {"LRG1": 1.4, "LRG2": 0.6, "QSO": 0.6}},
        "f_ic": { # incompleteness parameter which is a downsampling factor controlling the overall number density of the mock galaxies
            'Base': {"LRG1": 0.92, "LRG2": 0.89, "QSO": 0.041, "LRG3": 0.92, "LRG4": 0.19},
            'Ext': {"LRG1": 0.70, "LRG2": 0.62, "QSO": 0.019}},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 4
        func = lambda p: Zheng2005().Nc(logM-np.log10(self.h), logMmin=p['logM_cut'], sigmalogM=np.sqrt(2)*p['sigma']) * p['f_ic']
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):
        self.require(['sample'])
        func1 = lambda p: Zheng2005().Ns(10**logM/self.h, M0=p['kappa']*10**p['logM_cut'], M1 = 10**p['logM_1'], alpha=p['alpha'])
        if self.sample[:3]=='LRG': func = lambda p: func1(p) * self.Ncen(logM)(p)  # Eq 5
        elif self.sample[:3]=='QSO': func = func1 # Eq 6
        return lambda p={}: func(self.p0 | p)

    # def nc(self, ks, zs, logM):
    #     return lambda p={}: 1  # TODO: add velocity bias

    # def ns(self, ks, zs, logM):



class Kou2023(BaseHOD, Studies.Kou2023):  # Kou 2023, arxiv.org/abs/2211.07502
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
        finc = lambda p: More2015().finc(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        func = lambda p: Zheng2005().Nc(logM, logMmin=p['logM_min'], sigmalogM=p['sigma_logM'])*finc(p)
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 20
        func = lambda p: Zheng2005().Ns(10**logM, M0=10**p['logM_min'], M1=10**p['logM_1'], alpha=1) * self.Ncen(logM)(p)
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





class Linke2022(BaseHOD, Studies.Linke2022):  # arxiv.org/abs/2204.02418
    models = {'sample':['MS', 'KVG', 'fid'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    params ={
        # Best-fit parameters, Table 3
        'alpha_a': {'MS': {'r': 0.47, 'b': 0.1}, 'KVG': {'r': 0.34, 'b': 0.13}},  # for the maximum of〈N^a|m〉, gives the fraction of massive halos (m >> M^a_th) with a central galaxy from population a
        'sigma_a': {'MS': {'r': 0.55, 'b': 0.47}, 'KVG': {'r': 0.52, 'b': 0.47}},  # determines the transition of 〈N^a_cen | m〉from 0 to α^a
        'M_th_a': {'MS': {'r': 23.0, 'b': 1.19}, 'KVG': {'r': 15, 'b': 1.4}},  # halo mass below which we do not expect halos to contain galaxies, 1e11 Msol
        'beta_a': {'MS': {'r': 0.84, 'b': 0.73}, 'KVG': {'r': 0.88, 'b': 0.55}},
        'M_a': {'MS': {'r': 5.8, 'b': 32}, 'KVG': {'r': 3.6, 'b': 20}},  # 1e13 Msol
        'f_a': {'MS': {'r': 1.49, 'b': 0.88}, 'KVG': {'r': 1.27, 'b': 0.83}},  # factor in difference of concentration used in NFW from halo matter profile to average number density of satellite galaxies
    }
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 36
        func = lambda p: Zheng2005().Nc(logM, logMmin=np.log10(p['M_th_a']*1e11), sigmalogM=p['sigma_a']) * p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005().Ns(10**logM/self.h/self.h, M0=0, M1 = p['M_a']*1e13, alpha=p['beta_a']) * self.Ncen(logM)(p)/p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def ns_r(self, rs, zs, logMs):
        rs, zs, logMs = self.setdim(rs, zs, logMs)
        rdels, cdels = self.r200m(zs, logMs), self.c200m(zs, logMs)
        xs = rs*u.Mpc/rdels
        NFW_func = 1/(xs*(1/cdels+xs)**2)
        A_NFW = (1+cdels)/(np.log(1+cdels)*(1+cdels)-cdels)
        rhom = self.rhoc(zs)*self.Omega_m
        val = 200*rhom/3 *NFW_func * A_NFW/(10**logMs*u.Msun)
        return lambda **kwargs: val

    def ns(self, ks, zs, logMs):
        self.require(['rhoc', 'c200m', 'r200m'])
        fft = HaloModels.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)()
        val = fft.FFT3D(NFW)
        return lambda **kwargs: val




class Hadzhiyska2025B(BaseHOD, Studies.Hadzhiyska2025B):  #
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
        func = lambda p: Zheng2005().Nc(logM-np.log10(self.h), logMmin=p['logMcut'], sigmalogM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005().Ns(10**logM/self.h, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)
    

class Hadzhiyska2025A(BaseHOD, Studies.Hadzhiyska2025A):  #
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
        func = lambda p: Zheng2005().Nc(logM, logMmin=p['logMcut'], sigmalogM=2*p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 3
        logM = logM-np.log10(self.h)
        func = lambda p: Zheng2005().Ns(logM, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)
    





    # def GNFW_r(self, rs, zs, logMs, rdel, cdel):  # GNFW in real space
    #     rs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions
    #     xs0 = rs*u.Mpc/(rdel(zs, logMs)/cdel(zs, logMs))  # define scaled radius
    #     xs = lambda f_tr: xs0*f_tr  # apply truncation
    #     gnfw = lambda x, gamma, alpha, beta: 1/(x**gamma * (1+x)**alpha)**((beta-gamma)/alpha)  # define GNFW
    #     return lambda gamma=1, alpha=1, beta=3, f_tr=1: gnfw(xs(f_tr), gamma, alpha, beta)

    # def NFW_k(self, ks, zs, logMs, rdel, cdel):  # Default satellite distribution (FFT of NFW profile)
    #     ks, zs, logMs = self.setdim(ks, zs, logMs)  # Set proper dimensions
    #     qs0 = ks/u.Mpc*(rdel(zs, logMs)/cdel(zs, logMs))  # define scaled wavenumber
    #     qs = lambda f_tr: qs0/f_tr  # apply truncation
    #     cs0 = cdel(zs, logMs)  # concentration
    #     cs = lambda f_tr: cs0*f_tr  # apply truncation
    #     Si, Ci = lambda q: sici(q)[0], lambda q: sici(q)[1]  # Define sine/cosine integrals
    #     nfw = lambda c, q: (np.cos(q)*(Ci((1+c)*q)-Ci(qs)) + np.sin(q)*(Si((1+c)*q)-Si(q)) - np.sin(c*q)/(1+c*q))  # define NFW
    #     return 

    # def NFW_k(self, ks, rdels, cdels):  # Default satellite distribution (FFT of NFW profile)
    #     c = lambda L_trunc: cdels.value*L_trunc  # Apply truncation if given
    #     Si, Ci = lambda x: sici(x)[0], lambda x: sici(x)[1]  # Define sine/cosine integrals
    #     trigpart = lambda c: (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))
    #     return lambda L_trunc: trigpart(c(L_trunc))

    # def ns(self, ks, zs, logMs):  # Eq 8-10
    #     self.require(['rhoc', 'c200c', 'r200c'])
    #     Si, Ci = lambda q: sici(q)[0], lambda q: sici(q)[1]  # Define sine/cosine integrals
    #     ks, zs, logMs = self.setdim(ks, zs, logMs)
    #     rdels, cdels = self.r200c(zs, logMs), self.c200c(zs, logMs)
    #     qs = (ks/u.Mpc*rdels/cdels).value  # Define scaled wavenumber

    #     f_NFW = lambda x: (np.log(1+x)-x/(1+x))**(-1)
    #     NFW_trunc = lambda c: (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))

    #     prefac = (self.rhoc(0)*self.Omega_m)/(10**logMs*u.Msun) # density from halo mass over mean density at z=0
    #     func = lambda p: prefac* NFW_trunc(p['lambda']*cdels.value)*f_NFW(cdels*p['lambda'])
    #     return lambda p={}: func(self.p0 | p)

    # def ns_r(self, rs, zs, logMs):
    #     fft = FFTs.mcfit_package(rs=rs)
    #     NFW = self.ns(fft.ks, zs, logMs)
    #     func = lambda p: fft.IFFT3D(NFW(p))
    #     return lambda p={}: func(self.p0 | p)