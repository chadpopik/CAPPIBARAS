""""
Models for the dust emission from galaxies in the tSZ signal. 

"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import Models.Studies as Studies


class BaseDust:
    def __init__(self):
        pass


class Amodeo2021(BaseDust, Studies.Amodeo2021):  # arxiv.org/abs/2009.05558
    models = {'fitdata': ['Hr', 'ACTHr'],}  # Fit to Hershel or Act and Hershel
    params = {
        # Best-fit dust params, all estimated from Fig11a, # TODO: get more accurate
        'A_dust': {'Hr': 0.326, 'ACTHr': 0.363},  # amplitude of dust emission [kJy/sr]
        'T_dust': {'Hr': 20.7,  'ACTHr': 16.9},   # Dust temperature [K]
        'beta_dust':{'Hr': 1.13, 'ACTHr': 1.13},  # Dust spectral index
        'c_0': {'Hr': 5.00,'ACTHr': 6.046},  # Polynomial coefficient on x^0
        'c_1': {'Hr': -1.48, 'ACTHr': -1.88}, # Polynomial coefficient on x^1
        'c_2': {'Hr': 0.113, 'ACTHr': 0.148}, # Polynomial coefficient on x^2
        # Fixed dust params
        'z0': 0.55,  # redshift of the dust emitters, II.A.p1
        'nu0': ((c.c/(350*u.um)).to(u.GHz)).value,  # rest-frame frequency at which we normalize the dust emission, assumed from matching Fig11 I(v) plots
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        self.require(['fitdata'])
    
    # Polynominal dust fit to stacked profiles I(nu) [in jr/sr]
    def dustpoly(self, R, nu, **kwargs):
        x = lambda nu, p: ((nu)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        planck = lambda p: (np.exp(x(self.p0['nu0']*u.GHz, p))-1)/(np.exp(x(nu*(1+self.p0['z0']), p))-1)  # Planck function part
        amp = lambda p: p['A_dust']*u.kJy/u.sr*(nu*(1+self.p0['z0'])/(self.p0['nu0']*u.GHz))**(p['beta_dust']+3)  # Amplitude part
        poly = lambda p: p['c_0']+p['c_1']*R+p['c_2']*R**2  # Polynomial part
        dustfunc = lambda p: amp(p)*planck(p)*poly(p)  # Combine
        return lambda p={}: dustfunc(self.p0 | p)

    # Conversion of polynomial fit to uK arcmin^2
    def dust_uKarcmin(self, R, nu, **kwargs):
        x = ((nu)*c.h/c.k_B/(self.T_CMB*u.K)).decompose().value
        dB_dT = ((2*c.h*(nu)**3/c.c**2)).to(u.kJy) * x/self.T_CMB * np.exp(x)/(np.exp(x)-1)**2  # Planck function for unit conversion to K
        dustprof = lambda p: self.dustpoly(R, nu)(p)/dB_dT*1e6 * np.pi*R**2  # Also multipy by area of disc
        return lambda p={}: dustprof(self.p0 | p).value
    
    # Conversion of polynomial fit to uK arcmin^2, TODO 2
    def dust_y(self, R, nu, **kwargs):
        x = (c.h * nu / (c.k_B * self.T_CMB*u.K)).decompose().value
        fnu = x / np.tanh(x / 2.0) - 4.0
        return lambda p={}: self.dust_uKarcmin(R, nu)(self.p0 | p) / (fnu*self.T_CMB*1e6)