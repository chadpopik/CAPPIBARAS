"""
Rewrite of HaloModels.astropy_model with the inputs flipped: z is fixed at construction
(as self.z) instead of being passed into every function call.

All unit handling (stripping astropy units) happens once in __init__, so the functions
themselves just use self.z directly with no further unit handling.

z is reshaped in __init__ the same way z is reshaped in the pyccl and colossus rewrites,
so results from all three line up. z may be left as None; __init__ never fails because
it's missing, only whichever function actually needs it does.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import numpy as np
import astropy
import astropy.cosmology
import astropy.units as u


def unitinput(val, unit=u.dimensionless_unscaled):
    return (val if isinstance(val, u.Quantity) else val*unit).to(unit).value


class astropy_model:  # https://docs.astropy.org/en/stable/cosmology/index.html
    def __init__(self, z=None, Cosmology='Planck18', **cosmo_params):
        cosmo = getattr(astropy.cosmology, Cosmology.capitalize())
        self.cosmology = cosmo.clone(**(cosmo.parameters | cosmo_params))
        self.params = {(k[1:] if k.startswith('_') else k): v for k, v in self.cosmology.__dict__.items()}

        z = None if z is None else unitinput(z)
        self.z = None if z is None else np.array(z, ndmin=1)  # z axis

    def H(self):  # Hubble function [km/s/Mpc], from self.z
        return self.cosmology.H(self.z)

    def chi(self):  # comoving distance [Mpc], from self.z
        return self.cosmology.comoving_distance(self.z)

    def dA(self):  # angular diameter distance [Mpc], from self.z
        return self.cosmology.angular_diameter_distance(self.z)

    def dL(self):  # luminosity distance [Mpc], from self.z
        return self.cosmology.luminosity_distance(self.z)

    def rhoc(self):  # critical density [Msun/Mpc^3], from self.z
        return self.cosmology.critical_density(self.z).to(u.Msun/u.Mpc**3)

    def rhom(self):  # mean matter density [Msun/Mpc^3], from self.z
        return self.rhoc() * self.params['Om0']

    def Vcom(self):  # comoving volume [Mpc^3], from self.z
        return self.cosmology.comoving_volume(self.z)
