""""
Dust Models
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

class Amodeo2021(BaseDust):
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
    
    def dustpoly(self, thetas, nu, z=0.45, nu0=856):
        amp = lambda p: p['A_dust']*(nu*(1+z)/nu0)**(p['beta_dust']+3)
        poly = lambda p: p['c_0']+p['c_1']*thetas+p['c_2']*thetas**2
        x = lambda nu, p: ((nu*u.GHz)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        planck = lambda p: (np.exp(x(nu0, p))-1)/(np.exp(x(nu*(1+z), p))-1)
        dustfunc = lambda p: amp(p)*planck(p)*poly(p)
        return lambda p={}: dustfunc(self.p0 | p)

    def dust_uK(self, thetas, nu, z=0.45, nu0=856):
        x = lambda p: ((nu*u.GHz)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        dB_dT = lambda p: ((2*c.h*(nu*u.GHz)**3/c.c**2)).to(u.kJy).value * x(p)/p['T_dust'] * np.exp(x(p))/(np.exp(x(p))-1)**2
        dustprof = lambda p: self.dustpoly(thetas, nu, z, nu0)(p)/dB_dT(p)*1e6*np.pi*thetas**2
        return lambda p={}: dustprof(self.p0 | p)