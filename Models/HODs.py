"""
HOD models
"""

import numpy as np
import scipy
from scipy.special import erf, sici
import Models.FFTs as FFTs
import Models.Studies as Studies
import astropy.units as u


class BaseHOD:
    def Nc_Z05(self, logM, logMmin, sigma, f_inc=1):  # Base form for expected number of centrals per halo
        return (f_inc/2) * (1+erf((logM-logMmin)/sigma))

    def Ns_Z05(self, logM, logM0, logM1, alpha):  # Base form for expected number of satellites per halo
        M, M0, M1 = 10**logM, 10**logM0, 10**logM1
        return np.where(M>=M0, ((M-M0)/M1)**alpha, 0)

    def finc_CMASS(self, logM, alpha_inc, logM_inc):  # CMASS incompleteness function (More 2015, arxiv.org/abs/1407.1856)
        return np.clip(1+alpha_inc*(logM-logM_inc), 0, 1)

    def setdim(self, ks, zs, logMs):  # Set proper dimensions of rs, zs, Ms and define xs
        ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)
        return ks, zs, logMs

    def nc(self, ks, zs, logMs):  # Default central distribution [n density] (FFT of dirac delta=1)
        ncfunc = np.vectorize(lambda k: 1)
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        return lambda **kwargs: ncfunc(ks)/u.Mpc**3

    def A_NFW(self, c):
        return (np.log(1+c)-c/(1+c))**(-1)

    def GNFW_r(self, rs, zs, logMs, rdel, cdel):
        xs = rs*u.Mpc/(rdel(zs, logMs)/cdel(zs, logMs))
        return lambda gamma=1, alpha=1, beta=3: 1/(xs**gamma * (1+xs)**alpha)**((beta-gamma)/alpha)

    # def NFW_k(self, ks, rdels, cdels):  # Default satellite distribution (FFT of NFW profile)
    #     c = lambda L_trunc: cdels.value*L_trunc  # Apply truncation if given
    #     Si, Ci = lambda x: sici(x)[0], lambda x: sici(x)[1]  # Define sine/cosine integrals
    #     trigpart = lambda c: (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))
    #     return lambda L_trunc: trigpart(c(L_trunc))

    def NFW_k(self, ks, rdel, cdel):  # Default satellite distribution (FFT of NFW profile)
        Si, Ci = lambda q: sici(q)[0], lambda x: sici(q)[1]  # Define sine/cosine integrals
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        qs = ks/u.Mpc*(rdel(zs, logMs)/cdel(zs, logMs))
        (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))
        return (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))



class DESI_1P(BaseHOD, Studies.Yuan2023):  # Yuan 2023, arxiv.org/abs/2306.06314
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        
        if self.zbin[:3]=='QSO': self.Ns = self.Ns_QSO
        elif self.zbin[:3]=='LRG': self.Ns = self.Ns_LRG

    def Nc(self, logM):
        func = lambda p: self.Nc_Z05(logM, logMmin=p['logM_cut'], sigma=np.sqrt(2)*p['sigma']) * p['f_ic']
        return lambda p={}: func(self.p0 | p)

    def Ns_QSO(self, logM):
        func = lambda p: self.Ns_Z05(logM, logM0=np.log10(p['kappa'])+p['logM_cut'], logM1 = p['logM_1'], alpha=p['alpha'])
        return lambda p={}: func(self.p0 | p)
    
    def Ns_LRG(self, logM):
        func = lambda p: self.Ns_Z05(logM, logM0=np.log10(p['kappa'])+p['logM_cut'], logM1 = p['logM_1'], alpha=p['alpha']) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)
    
    
    
class BOSS_DR12(BaseHOD, Studies.Kou2023):  # Kou 2023, arxiv.org/abs/2211.07502
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

    def Nc(self, logM):
        finc = lambda p: self.finc_CMASS(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        func = lambda p: self.Nc_Z05(logM, logMmin=p['logM_min'], sigma=p['sigma_logM'], f_inc=finc(p))
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        func = lambda p: self.Ns_Z05(logM, logM0=p['logM_min'], logM1=p['logM_1'], alpha=1) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)
    
    def ns_r(self, rs, zs, logMs):
        rs, zs, logMs = self.setdim(rs, zs, logMs)
        GNFW_func = lambda p: self.GNFW_r(rs, zs, logMs, self.r200m, self.c200m)(beta=p['beta_s'])
        func = lambda p: GNFW_func(p)/np.trapz(GNFW_func(p)*4*np.pi*rs**2, rs, axis=0)
        return lambda p={}: func(self.p0 | p)

        # rs, zs, logMs = self.setdim(rs, zs, logMs)
        # xs = rs*u.Mpc/(self.r200m(zs, logMs)/self.c200m(zs, logMs))  # scaled radius
        # NFW = 1 / (xs * (1+xs)**2)
        # NFW_int = np.trapz(NFW, xs, axis=0)
        # print(np.trapz(NFW, xs, axis=0), np.trapz(NFW, rs, axis=0))
        # return lambda p={}: NFW/NFW_int
    
    def ns(self, ks, zs, logMs):  # Default satellite distribution (FFT of NFW)
        self.require(['c200m', 'r200m'])
        fft = FFTs.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)
        return lambda p={}: fft.FFT3D(NFW(p))
    
    


class unWISE(BaseHOD, Studies.Kusiak2022):  # Kusiak 2022, arxiv.org/abs/2203.12583
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
    
    def Nc(self, logMs):
        func = lambda p: self.Nc_Z05(logMs, logMmin=p['logM_min^HOD'], sigma=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logMs):
        func = lambda p: self.Ns_Z05(logMs, logM0=-np.inf, logM1=p['logM_1'], alpha=p['alpha_s']) * self.Nc(logMs)(p)
        return lambda p={}: func(self.p0 | p)
    
    def ns(self, ks, zs, logMs):  # Eq 8-10
        self.require(['rhoc', 'c200c', 'r200c'])
        Si, Ci = lambda q: sici(q)[0], lambda q: sici(q)[1]  # Define sine/cosine integrals
        ks, zs, logMs = self.setdim(ks, zs, logMs)
        rdels, cdels = self.r200c(zs, logMs), self.c200c(zs, logMs)
        qs = (ks/u.Mpc*rdels/cdels).value  # Define scaled wavenumber
        
        f_NFW = lambda x: (np.log(1+x)-x/(1+x))**(-1)
        NFW_trunc = lambda c: (np.cos(qs) * (Ci((1+c)*qs)-Ci(qs)) + np.sin(qs) * (Si((1+c)*qs)-Si(qs)) - np.sin(c*qs)/(1+c*qs))
        
        prefac = (self.rhoc(0)*self.Omega_m)/(10**logMs*u.Msun) # density from halo mass over mean density at z=0
        func = lambda p: prefac* NFW_trunc(p['lambda']*cdels.value)*f_NFW(cdels*p['lambda'])
        return lambda p={}: func(self.p0 | p)
    
    def ns_r(self, rs, zs, logMs):
        fft = FFTs.mcfit_package(rs=rs)
        NFW = self.ns(fft.ks, zs, logMs)
        func = lambda p: fft.IFFT3D(NFW(p))
        return lambda p={}: func(self.p0 | p)


    
    
class KV450xGAMA(BaseHOD, Studies.Linke2022):  # Linke 2022, arxiv.org/abs/2204.02418
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        
    def Nc(self, logM):
        func = lambda p: self.Nc_Z05(logM, logMmin=np.log10(p['M_th^a']*1e11), sigma=p['sigma^a']) * p['alpha^a']
        return lambda p={}: func(self.p0 | p)
    
    def Ns(self, logM):
        func = lambda p: self.Ns_Z05(logM, logM0=0, logM1 = np.log10(p['M^a']*1e13), alpha=p['beta^a']) * self.Nc_Z05(logM, logMmin=np.log10(p['M_th^a']*1e11), sigma=p['sigma^a'])
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
        fft = FFTs.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)()
        val = fft.FFT3D(NFW)
        return lambda **kwargs: val
    


class BOSS_DR11(BaseHOD, Studies.More2015):  # More 2015, arxiv.org/abs/1407.1856
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

    def Nc(self, logM):
        finc = lambda p: self.finc_CMASS(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        func = lambda p: self.Nc_Z05(logM, logMmin=p['logM_min'], sigma=p['sigma^2']**0.5, f_inc=finc(p))
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        func = lambda p: self.Ns_Z05(logM, logM0=np.log10(p['kappa'])+p['logM_min'], logM1 = p['logM_1'], alpha=p['alpha']) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)