"""
Rewrite of HaloModels.colossus_model. z, logM, k, and R are passed into every function
call rather than being fixed at construction, same as the model choices (MassFunc,
HaloBias, Concentration, MassDef, PlinMod).

All unit handling (stripping astropy units, converting to the h-scaled units colossus
expects) happens inside each function, so the caller can pass z/logM/k/R as bare
numbers/arrays or as astropy Quantities.

z/logM/k/R are also reshaped onto separate broadcasting axes inside each function, so that
any function combining two or more of them (e.g. bh(z, M), Plin(k, z)) naturally comes out
shaped (n_r_or_k, n_z, n_M) rather than requiring the caller to line up matching lengths.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import numpy as np
import astropy.units as u
from colossus.cosmology import cosmology, power_spectrum
from colossus.halo import concentration, mass_defs, mass_adv, profile_nfw
from colossus.lss import bias, mass_function


class Cosmology:
    def __init__(self, Cosmology='planck18', **cosmo_params):
        self.Cosmology = Cosmology if Cosmology[0:3].lower() in ['wma', 'eds'] else Cosmology.lower()
        self.cosmology = cosmology.setCosmology(self.Cosmology, cosmo_params)

    def _reshape_z(self, z):  # plain z, natural shape preserved (scalar stays scalar)
        return None if z is None else (z.to(u.dimensionless_unscaled).value if isinstance(z, u.Quantity) else z)

    def H(self, z):  # Hubble Function [km/s/Mpc]
        z = self._reshape_z(z)
        return self.cosmology.Hz(z=z) *u.km/u.s/u.Mpc

    # colossus's comovingDistance integrates one z at a time internally and needs a flat z array, unlike the other z-only functions
    def chi(self, z):  # Comoving Distance [Mpc]
        z = np.array(self._reshape_z(z), ndmin=1)
        return self.cosmology.comovingDistance(z_min=0.0, z_max=z.flatten()) *u.Mpc/self.cosmology.h

    def dA(self, z):  # Angular Distance [Mpc]
        z = self._reshape_z(z)
        return self.cosmology.angularDiameterDistance(z=z) *u.Mpc/self.cosmology.h

    def Vcom(self, z):  # Comoving volume [Mpc^3]
        return 4*np.pi/3 * self.chi(z)**3

    def rhoc(self, z):  # Critical Density [Msol/kpc^3*h^2 -> Msol/Mpc^3]
        z = self._reshape_z(z)
        return (self.cosmology.rho_c(z=z) *u.Msun/u.kpc**3*self.cosmology.h**2).to(u.Msun/u.Mpc**3)

    def rhom(self, z):  # Mean Matter Density [Msol/Mpc^3]
        return self.rhoc(z) * self.cosmology.Om0


class HaloModel(Cosmology):
    def __init__(self, Cosmology='planck18', **cosmo_params):
        super().__init__(Cosmology=Cosmology, **cosmo_params)

    def _reshape_z_axis(self, z):  # z axis (middle), for combining with logM/k/R
        z = self._reshape_z(z)
        return None if z is None else np.array(z, ndmin=1)[:, None]

    def _reshape_logM(self, logM):  # M axis (last)
        logM = logM.to(u.dimensionless_unscaled).value if isinstance(logM, u.Quantity) else logM
        return None if logM is None else np.array(logM, ndmin=1)

    def _reshape_k(self, k):  # k axis (first)
        k = None if k is None else ((k/self.cosmology.h).to(u.Mpc**-1).value if isinstance(k, u.Quantity) else k/self.cosmology.h)
        return None if k is None else np.array(k, ndmin=1)[:, None, None]

    def _reshape_R(self, R):  # r axis (first)
        R = None if R is None else R.to(u.kpc).value*self.cosmology.h
        return None if R is None else np.array(R, ndmin=1)[:, None, None]

    # colossus only broadcasts k/z pairs of matching shape, so this needs to loop to combine k and z onto separate axes
    def Plin(self, k, z, PlinMod):  # Linear Matter Power Spectrum [h^3/Mpc^3 --> Msol/Mpc^3]
        k = self._reshape_k(k)
        z = self._reshape_z_axis(z)
        Plinfunc = lambda k, z: self.cosmology.matterPowerSpectrum(k=k, z=z, model=PlinMod)
        return np.vectorize(Plinfunc)(k, z) /self.cosmology.h**3 *u.Mpc**3

    # colossus only broadcasts z/M pairs of matching shape, so this needs to loop to combine z and logM onto separate axes
    def bh(self, z, logM, HaloBias, MassDef):  # Halo Bias []
        z = self._reshape_z_axis(z)
        logM = self._reshape_logM(logM)
        bhfunc = lambda z, M: bias.haloBias(M, z=z, model=HaloBias, mdef=MassDef)
        return np.vectorize(bhfunc)(z, 10**logM*self.cosmology.h) *u.dimensionless_unscaled

    # some MassFunc models (e.g. tinker08, watson13) don't support array z internally, so this still needs to loop over halos
    def dndlogm(self, z, logM, MassFunc, MassDef):  # Halo Mass Function [1/Mpc^3/dex]
        z = self._reshape_z_axis(z)
        logM = self._reshape_logM(logM)
        dndlogmfunc = lambda z, M: np.vectorize(mass_function.massFunction)(M, z=z, model=MassFunc, mdef=MassDef, q_out='dndlnM')
        M_h = 10**logM*self.cosmology.h
        dlogMfac = (np.log(M_h[1])-np.log(M_h[0]))/(np.log10(M_h[1])-np.log10(M_h[0]))
        return dndlogmfunc(z, M_h) *dlogMfac *self.cosmology.h**3 *1/u.Mpc**3/u.dex

    # colossus builds one NFWProfile object per halo, so unlike the other functions this can't take array z/M/c directly and still needs to loop over halos
    def NFW_r(self, R, z, logM, MassDef, Concentration):  # NFW density profile
        R = self._reshape_R(R)
        z = self._reshape_z_axis(z)
        logM = self._reshape_logM(logM)
        NFWfunc = lambda r, z, M: profile_nfw.NFWProfile(z=z, M=M*self.cosmology.h, c=concentration.concentration(M*self.cosmology.h, z=z, model=Concentration, mdef=MassDef), mdef=MassDef).density(r=r)
        return (np.vectorize(NFWfunc)(R, z, 10**logM) *u.Msun*self.cosmology.h**2/u.kpc**3).to(u.Msun/u.Mpc**3)

    # some Concentration models (e.g. bullock01, bhattacharya13) don't support array z internally, so this still needs to loop over halos
    def c(self, z, logM, Concentration, MassDef):  # Concentration []
        z = self._reshape_z_axis(z)
        logM = self._reshape_logM(logM)
        cfunc = lambda z, M: concentration.concentration(M, z=z, model=Concentration, mdef=MassDef)
        return np.vectorize(cfunc)(z, 10**logM*self.cosmology.h) *u.dimensionless_unscaled

    # colossus's changeMassDefinition requires a scalar redshift, so unlike the other functions this still needs to loop over halos
    def logM_conv(self, z, logM, newmdef, MassDef, Concentration, conc=None):  # Convert mass definition
        z_r = self._reshape_z_axis(z)
        logM_r = self._reshape_logM(logM)
        Mconvattr = mass_defs.changeMassDefinition
        if conc is not None:
            Mconvfunc = lambda zz, MM, conc: np.vectorize(Mconvattr)(10**MM*self.cosmology.h, conc(zz, MM), zz, MassDef, newmdef)
            return lambda conc: np.log10(Mconvfunc(z_r, logM_r, conc)[0])
        else:
            # self.c() is passed the raw z/logM (not z_r/logM_r) so it applies its own reshape and lines up with z_r/logM_r's broadcast shape
            Mconvfunc = lambda zz, MM: np.vectorize(Mconvattr)(10**MM*self.cosmology.h, self.c(z, logM, Concentration, MassDef).value, zz, MassDef, newmdef)
            return np.log10(Mconvfunc(z_r, logM_r)[0])
