""""
Models for the dust emission from galaxies in the tSZ signal. 

TODO 1: Get more accurate values for the parameters in Amodeo 2021
TODO 2: Figure out how to use dust without frequency dependence (DR6 ILC)
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import Models.Studies as Studies


class BaseDust:
    def setup(self, study, inputs, **kwargs):   # Grab user defined attributes from Studies module
        info = getattr(Studies, study)(inputs, **kwargs)
        for attr in dir(info):
            if attr[:2]!='__': setattr(self, attr, getattr(info, attr))
        self.require = lambda reqlist, reqdict=None: info.check_inputs(inputs | kwargs, subdict=reqdict, sublist=reqlist)


class Amodeo2021(BaseDust):  # arxiv.org/abs/2009.05558
    def __init__(self, inputs, **kwargs):
        self.setup('Amodeo2021Dust', inputs, **kwargs)
    
    # Polynominal dust fit to stacked profiles I(nu) [in jr/sr]
    def dustpoly(self, R, nu, **kwargs):
        x = lambda nu, p: ((nu*u.GHz)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        planck = lambda p: (np.exp(x(self.nu0, p))-1)/(np.exp(x(nu*(1+self.z0), p))-1)  # Planck function part
        amp = lambda p: p['A_dust']*(nu*(1+self.z0)/self.nu0)**(p['beta_dust']+3)  # Amplitude part
        poly = lambda p: p['c_0']+p['c_1']*R+p['c_2']*R**2  # Polynomial part
        dustfunc = lambda p: amp(p)*planck(p)*poly(p)  # Combine
        return lambda p={}: dustfunc(self.p0 | p).value

    # Conversion of polynomial fit to uK arcmin^2
    def dust_uKarcmin(self, R, nu, **kwargs):
        x = ((nu*u.GHz)*c.h/c.k_B/(self.T_CMB*u.K)).decompose().value
        dB_dT = ((2*c.h*(nu*u.GHz)**3/c.c**2)).to(u.kJy) * x/self.T_CMB * np.exp(x)/(np.exp(x)-1)**2  # Planck function for unit conversion to K
        dustprof = lambda p: self.dustpoly(R, nu)(p)/dB_dT*1e6 * np.pi*R**2  # Also multipy by area of disc
        return lambda p={}: dustprof(self.p0 | p).value
    
    # Conversion of polynomial fit to uK arcmin^2, TODO 2
    def dust_y(self, R, nu, **kwargs):
        x = (c.h * nu*u.GHz / (c.k_B * self.T_CMB*u.K)).decompose().value
        fnu = x / np.tanh(x / 2.0) - 4.0
        return lambda p={}: self.dust_uKarcmin(R, nu)(self.p0 | p) / (fnu*self.T_CMB*1e6)