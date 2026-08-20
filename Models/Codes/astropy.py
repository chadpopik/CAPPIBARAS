"""
Rewrite of HaloModels.astropy_model. Unlike the pyccl and colossus rewrites, z is passed
into every function call rather than being fixed at construction.

All unit handling (stripping astropy units) happens inside each function, so the caller
can pass z as a bare number/array or as an astropy Quantity.

z is reshaped the same way z is reshaped in the pyccl and colossus rewrites, so results
from all three line up.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import numpy as np
import astropy
import astropy.cosmology
import astropy.units as u


def unitinput(val, unit=u.dimensionless_unscaled):
    return (val if isinstance(val, u.Quantity) else val*unit).to(unit).value


class Cosmology:  # https://docs.astropy.org/en/stable/cosmology/index.html
    # Available preset cosmologies in astropy
    available_cosmologies = list(astropy.cosmology.available)
    # Available cosmo parameters to set in astropy cosmology 
    cosmo_input_params = list(astropy.cosmology.Planck18.parameters.keys())
    
    def __init__(self, cosmo_name, **cosmo_params):
        # Check if cosmo_name is in the list of available cosmologies
        if cosmo_name not in self.available_cosmologies:
            raise ValueError(f"Choose cosmology in {self.available_cosmologies}")
        
        # Get default cosmology and then update based on input cosmological parameters
        cosmo_default = getattr(astropy.cosmology, cosmo_name)
        self.cosmology = cosmo_default.clone(**(cosmo_default.parameters | cosmo_params))
        
        
        self.params = {(k[1:] if k.startswith('_') else k): v for k, v in self.cosmology.__dict__.items()}

    def _reshape_z(self, z):  # z axis
        return None if z is None else np.array(unitinput(z), ndmin=1)

    def H(self, z):  # Hubble function [km/s/Mpc]
        z = self._reshape_z(z)
        return self.cosmology.H(z)

    def chi(self, z):  # comoving distance [Mpc]
        z = self._reshape_z(z)
        return self.cosmology.comoving_distance(z)

    def dA(self, z):  # angular diameter distance [Mpc]
        z = self._reshape_z(z)
        return self.cosmology.angular_diameter_distance(z)

    def dL(self, z):  # luminosity distance [Mpc]
        z = self._reshape_z(z)
        return self.cosmology.luminosity_distance(z)

    def rhoc(self, z):  # critical density [Msun/Mpc^3]
        z = self._reshape_z(z)
        return self.cosmology.critical_density(z).to(u.Msun/u.Mpc**3)

    def rhom(self, z):  # mean matter density [Msun/Mpc^3]
        return self.rhoc(z) * self.params['Om0']

    def Vcom(self, z):  # comoving volume [Mpc^3]
        z = self._reshape_z(z)
        return self.cosmology.comoving_volume(z)
