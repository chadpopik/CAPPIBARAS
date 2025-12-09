"""
Collections of radial halo profiles used to forward model SZ signals, specifically thermal pressure and gas density.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import Models.FFTs as FFTs
import Models.Studies as Studies


class BaseProfile:
    def MGNFW(self, x, P0, xc, gamma, alpha, beta):  # Modified GNFW used for pressure profile in B11
        return P0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)

    def GNFW(self, x, rho0, xc, gamma, alpha, beta):  # GNFW used for density profile in B18
        return rho0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-(beta-gamma)/alpha)

    def PLmz(self, z, logm200c, A0, alpham, alphaz):  # Form of Mh and z dependance of GNFW parameters in B1
        return A0 * (10**logm200c/1.e14)**alpham * (1.+z)**alphaz

    def unitfac(self, prof, units='cosmo'):  # handles units of rho and pth for cosmo and cgs
        if units=='cosmo': return 1  # cosmological units are the default
        elif units=='cgs':  # if cgs, convert from cosmo units to cgs
            if prof=='pres': return (1*u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)/(u.Msun/u.Mpc/u.s**2)
            if prof=='dens': return (1*u.Msun/u.Mpc**3).to(u.g/u.cm**3)/(u.Msun/u.Mpc**3)
        
    def setdim(self, rs, zs, logMs):  # Set proper dimensions of rs, zs, Ms (and define xs)
        rs, zs, logMs = np.array(rs, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)
        return rs*u.Mpc/self.r200c(zs, logMs), zs, logMs

    def twohalo(self, rs, zs, logMs, logMs_2h, windowfunc=lambda k: 1): # Two-halo component calculated with linear theory
        fft = FFTs.mcfit_package(rs=rs)  # setup FFT
        ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions

        ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)  # Assign proper dimensions [nr, nz, nm]
        
        prefac = self.bh(zs, logMs)*self.Plin(ks, zs)*windowfunc(ks)  # collect factors outside int
        intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
        P2h = lambda prof1h: prefac*(np.trapz(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
        return lambda prof1h: IFFT3D(P2h(prof1h)) *prof1h.unit  # IFFT to real space and return its units destroyed by the FFT


    
class Kou2023(BaseProfile, Studies.Kou2023):  # TODO: in progress
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
    def Pe(self):
        fac1 = 1.65*(h/0.7)**2 * u.eV*u.cm**3
        fac2 = E(z)**(8/3) * (p['1-bh']*M500c/(3e14*(0.7/h)*u.Msun))**(2/3+p['alpha_p'])
        rs = r200m/c
        fac3 = self.GNFW(r/rs, rho0=6.41, xc=1, gamma=0.31, alpha=1.33, beta=4.13)



class Amodeo2021(BaseProfile, Studies.Amodeo2021):
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def rho1h(self, rs, zs, logMs, units='cosmo'):  # one-halo density component, II.C.eq16
        self.require(['r200c', 'rhoc'])  # required functions
        xs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]        
        factorfront = self.rhoc(zs)*self.f_bary *self.unitfac('dens', units)  # group all fixed values and convert units 
        
        func = lambda p: self.GNFW(xs, gamma=self.gamma_k, alpha=self.alpha_k, rho0=10**p['logrho0'], xc=p['xc_k'], beta=p['beta_k'])  # free parameterization
        return lambda p={}: factorfront*func(self.p0 | p)

    def Pth1h(self, rs, zs, logMs, units='cosmo'): # one-halo pressure component, Section II.C Eq 17
        self.require(['r200c', 'rhoc'])  # required functions
        xs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]        
        Ms, G_cosmo = 10**logMs *u.Msun, c.G.to(u.Mpc**3/u.Msun/u.s**2)
        Ps200c = G_cosmo * Ms*200*self.rhoc(zs)/(2*self.r200c(zs, logMs))  # scale pressure
        factorfront = Ps200c*self.f_bary *self.unitfac('pres', units)  # group all fixed values and convert units
        
        xc = self.PLmz(zs, logMs, A0=self.xc_t_A0, alpham=self.xc_t_alpham, alphaz=self.xc_t_alphaz)
        func = lambda p: self.MGNFW(xs, gamma=self.gamma_t, alpha=p['alpha_t'], P0 = p['P0'], xc=xc, beta=p['beta_t'])
        return lambda p={}: factorfront*func(self.p0 | p)

    def prof2h(self, rs, zs, logMs): # linear two-halo calculation, Section II.C Eq 17
        self.require(['r200c', 'rhoc', 'dndlogm', 'bh', 'Plin'])  # required functions
        windfunc = lambda k: np.where(k > 1/50, 1, 0)  # two-halo window function, [k]=1/Mpc
        logMs_2h = np.linspace(self.info['logmhalomin_2h'], self.info['logmhalomax_2h'], 50)
        lin2h = self.twohalo(rs, zs, logMs, logMs_2h, windowfunc=windfunc)  # linear two-halo calculation
        return lambda prof, p={}: lin2h(prof(rs, zs, logMs_2h)(p))
    
    def rho2h(self, rs, zs, logMs, units='cosmo'): # two-halo density component
        rho1h = Battaglia2018({'model':'AGN'}, rhoc=self.rhoc, r200c=self.r200c).rho1h
        lin2hrho = self.prof2h(rs, zs, logMs)(rho1h) *self.unitfac('dens', units)
        return lambda p={}: lin2hrho
    
    def Pth2h(self, rs, zs, logMs, units='cosmo'): # two-halo pressure component
        pth1h = Battaglia2011(rhoc=self.rhoc, r200c=self.r200c).Pth1h
        lin2hPth = self.prof2h(rs, zs, logMs)(pth1h) *self.unitfac('pres', units)
        return lambda p={}: lin2hPth
    
    def Pth(self, rs, zs, logMs, units='cosmo'):
        Pth_1h, Pth_2h = self.Pth1h(rs, zs, logMs, units), self.Pth2h(rs, zs, logMs, units)
        Pth = lambda p={}: Pth_1h(p)+p['A2h_t']*Pth_2h(p)
        return lambda p={}: Pth(self.p0 | p)
    
    def rho(self, rs, zs, logMs, units='cosmo'):
        rho_1h, rho_2h = self.rho1h(rs, zs, logMs, units), self.rho2h(rs, zs, logMs, units)
        rho = lambda p={}: rho_1h(p)+p['A2h_k']*rho_2h(p)
        return lambda p={}: rho(self.p0 | p)



class Battaglia2018(BaseProfile, Studies.Battaglia2018):
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def rho1h(self, rs, zs, logMs, units='cosmo'):  # B18 eq A1
        self.require(['rhoc', 'r200c'])
        xs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        factorfront = self.rhoc(zs)*self.f_bary *self.unitfac('dens', units)  # prefactor and units
        rho_rhodel = lambda p: self.GNFW(xs, gamma=self.gamma, xc=self.xc,
                                    alpha=self.PLmz(zs, logMs, p['alpha_A0'], p['alpha_alpham'], p['alpha_alphaz']),
                                    rho0=self.PLmz(zs, logMs, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
                                    beta=self.PLmz(zs, logMs, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*rho_rhodel(self.p0 | p)


class Battaglia2011(BaseProfile, Studies.Battaglia2011):
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def Pth1h(self, rs, zs, logMs, units='cosmo'):  # B18 Eq. A1
        self.require(['rhoc', 'r200c'])
        xs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        Ms, G_cosmo = 10**logMs *u.Msun, c.G.to(u.Mpc**3/u.Msun/u.s**2)  # halo masses and G constant in cosmo units
        p200c = G_cosmo*Ms*200*self.rhoc(zs)/(2*self.r200c(zs, logMs))  # Scaled pressure of 200c sphere
        factorfront = p200c*self.f_bary *self.unitfac('pres', units)  # combined prefactor and units
        Pth_Pdel = lambda p: self.MGNFW(xs, gamma=self.gamma_pres, alpha=self.alpha_pres,  # set parameterization
                                    P0=self.PLmz(zs, logMs, p['P0_A0'], p['P0_alpham'], p['P0_alphaz']),
                                    xc=self.PLmz(zs, logMs, p['xc_A0'], p['xc_alpham'], p['xc_alphaz']),
                                    beta=self.PLmz(zs, logMs, p['beta_pres_A0'], p['beta_pres_alpham'], p['beta_pres_alphaz']))
        return lambda p={}: factorfront*Pth_Pdel(self.p0 | p)