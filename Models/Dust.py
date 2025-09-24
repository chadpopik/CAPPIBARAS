""""
Models for the dust emission from galaxies in the tSZ signal. 

TODO 1: Get more accurate values for the parameters in Amodeo 2021
"""

import numpy as np
import astropy.units as u
import astropy.constants as c

class BaseDust:
    # Check validitity of model specification and assign default parameters
    def checkspefs(self, spefs, required):
        for spef in required:
            if spefs[spef] not in getattr(self, f"{spef}s"):
                raise NameError(f"{spef} {spefs[spef]} doesn't exist, choose from available {spef}s: {getattr(self, f'{spef}s')}")
            else:
                setattr(self, spef, spefs[spef])
        self.p0 = {param: self.params[param][self.models.index(self.model)] for param in self.params.keys()}

class Amodeo2021(BaseDust):  # (Amodeo+ 2021, arxiv.org/abs/2009.05558)
    info = {'z': 0.45,  # redshift of the dust emitters
            'nu0': 856.54988, # rest-frame frequency at which we normalize the dust emission
            }
    # TODO 1: These are all estimated values from looking at the paper, don't have the actual best fit value
    models=['Hershel', 'ACT+Hershel']
    params = {'A_dust': [0.326, 0.363],  # amplitude of dust emission [kJy/sr]
              'T_dust': [20.7, 16.9],  # Dust temperature [K]
              'beta_dust': [1.13, 1.13],  # Dust sepctral index
              'c_0': [5.00, 6.046],  # Polynomial coefficient on x^0
              'c_1': [-1.48, -1.88],   # Polynomial coefficient on x^1
              'c_2': [0.113, 0.148],  # Polynomial coefficient on x^2
            }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])
    
    # Polynominal dust fit to stacked profiles I(nu) [in jr/sr]
    def dustpoly(self, thetas, nu, z=0.55, nu0=856.54988):
        x = lambda nu, p: ((nu*u.GHz)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        planck = lambda p: (np.exp(x(nu0, p))-1)/(np.exp(x(nu*(1+z), p))-1)  # Planck function part
        amp = lambda p: p['A_dust']*(nu*(1+z)/nu0)**(p['beta_dust']+3)  # Amplitude part
        poly = lambda p: p['c_0']+p['c_1']*thetas+p['c_2']*thetas**2  # Polynomial part
        dustfunc = lambda p: amp(p)*planck(p)*poly(p)  # Combine
        return lambda p={}: dustfunc(self.p0 | p)

    # Conversion of polynomial fit to uK arcmin^2
    def dust_uKarcmin(self, thetas, nu, z=0.55, nu0=856.54988, T_CMB=2.7255):
        x = ((nu*u.GHz)*c.h/c.k_B/(T_CMB*u.K)).decompose()
        dB_dT = ((2*c.h*(nu*u.GHz)**3/c.c**2)).to(u.kJy).value * x/T_CMB * np.exp(x)/(np.exp(x)-1)**2  # Planck function for unit conversion to K
        dustprof = lambda p: self.dustpoly(thetas, nu, z, nu0)(p)/dB_dT*1e6 * np.pi*thetas**2  # Also multipy by area of disc
        return lambda p={}: dustprof(self.p0 | p)