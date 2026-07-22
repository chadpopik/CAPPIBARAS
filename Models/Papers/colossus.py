"""
Rewrite of HaloModels.colossus_model with the inputs flipped: z, logM, k, and R are
fixed at construction (as self.z, self.logM, self.k, self.R) instead of being passed
into every function call, while the model choices (MassFunc, HaloBias, Concentration,
MassDef, PlinMod) are now passed into each function call instead of being fixed at
construction.

All unit handling (stripping astropy units, converting to the h-scaled units colossus
expects) happens once in __init__, so the functions themselves just use self.z/self.logM
directly with no further unit handling.

z/logM/k/R are also reshaped onto separate broadcasting axes in __init__, so that any
function combining two or more of them (e.g. bh(z, M), Plin(k, z)) naturally comes out
shaped (n_r_or_k, n_z, n_M) rather than requiring the caller to line up matching lengths.
Any of z/logM/k/R may be left as None; __init__ never fails because one is missing, only
whichever function actually needs it does.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import numpy as np
import astropy.units as u
from colossus.cosmology import cosmology, power_spectrum
from colossus.halo import concentration, mass_defs, mass_adv, profile_nfw
from colossus.lss import bias, mass_function


class colossus_model:
    def __init__(self, z=None, logM=None, k=None, R=None, Cosmology='planck18', **cosmo_params):
        self.Cosmology = Cosmology if Cosmology[0:3].lower() in ['wma', 'eds'] else Cosmology.lower()
        self.cosmology = cosmology.setCosmology(self.Cosmology, cosmo_params)

        z = z.to(u.dimensionless_unscaled).value if isinstance(z, u.Quantity) else z
        self.z = None if z is None else np.array(z, ndmin=1)[:, None]  # z axis (middle)

        logM = logM.to(u.dimensionless_unscaled).value if isinstance(logM, u.Quantity) else logM
        self.logM = None if logM is None else np.array(logM, ndmin=1)  # M axis (last)

        k = None if k is None else ((k/self.cosmology.h).to(u.Mpc**-1).value if isinstance(k, u.Quantity) else k/self.cosmology.h)
        self.k = None if k is None else np.array(k, ndmin=1)[:, None, None]  # k axis (first)

        R = None if R is None else R.to(u.kpc).value*self.cosmology.h
        self.R = None if R is None else np.array(R, ndmin=1)[:, None, None]  # r axis (first)

    def H(self):  # Hubble Function [km/s/Mpc], from self.z
        return self.cosmology.Hz(z=self.z) *u.km/u.s/u.Mpc

    # colossus's comovingDistance integrates one z at a time internally and needs a flat z array, unlike the other z-only functions
    def chi(self):  # Comoving Distance [Mpc], from self.z
        return self.cosmology.comovingDistance(z_min=0.0, z_max=self.z.flatten())[:, None] *u.Mpc/self.cosmology.h

    def dA(self):  # Angular Distance [Mpc], from self.z
        return self.cosmology.angularDiameterDistance(z=self.z) *u.Mpc/self.cosmology.h

    def Vcom(self):  # Comoving volume [Mpc^3], from self.z
        return 4*np.pi/3 * self.chi()**3

    def rhoc(self):  # Critical Density [Msol/kpc^3*h^2 -> Msol/Mpc^3], from self.z
        return (self.cosmology.rho_c(z=self.z) *u.Msun/u.kpc**3*self.cosmology.h**2).to(u.Msun/u.Mpc**3)

    def rhom(self):  # Mean Matter Density [Msol/Mpc^3], from self.z
        return self.rhoc() * self.cosmology.Om0

    # colossus only broadcasts k/z pairs of matching shape, so this needs to loop to combine self.k and self.z onto separate axes
    def Plin(self, PlinMod):  # Linear Matter Power Spectrum [h^3/Mpc^3 --> Msol/Mpc^3], from self.k and self.z
        Plinfunc = lambda k, z: self.cosmology.matterPowerSpectrum(k=k, z=z, model=PlinMod)
        return np.vectorize(Plinfunc)(self.k, self.z) /self.cosmology.h**3 *u.Mpc**3

    # colossus only broadcasts z/M pairs of matching shape, so this needs to loop to combine self.z and self.logM onto separate axes
    def bh(self, HaloBias, MassDef):  # Halo Bias [], from self.z and self.logM
        bhfunc = lambda z, M: bias.haloBias(M, z=z, model=HaloBias, mdef=MassDef)
        return np.vectorize(bhfunc)(self.z, 10**self.logM*self.cosmology.h) *u.dimensionless_unscaled

    # some MassFunc models (e.g. tinker08, watson13) don't support array z internally, so this still needs to loop over halos
    def dndlogm(self, MassFunc, MassDef):  # Halo Mass Function [1/Mpc^3/dex], from self.z and self.logM
        dndlogmfunc = lambda z, M: np.vectorize(mass_function.massFunction)(M, z=z, model=MassFunc, mdef=MassDef, q_out='dndlnM')
        M_h = 10**self.logM*self.cosmology.h
        dlogMfac = (np.log(M_h[1])-np.log(M_h[0]))/(np.log10(M_h[1])-np.log10(M_h[0]))
        return dndlogmfunc(self.z, M_h) *dlogMfac *self.cosmology.h**3 *1/u.Mpc**3/u.dex

    # colossus builds one NFWProfile object per halo, so unlike the other functions this can't take array z/M/c directly and still needs to loop over halos
    def NFW_r(self, MassDef, Concentration):  # NFW density profile, from self.R, self.z, self.logM
        NFWfunc = lambda r, z, M: profile_nfw.NFWProfile(z=z, M=M*self.cosmology.h, c=concentration.concentration(M*self.cosmology.h, z=z, model=Concentration, mdef=MassDef), mdef=MassDef).density(r=r)
        return (np.vectorize(NFWfunc)(self.R, self.z, 10**self.logM) *u.Msun*self.cosmology.h**2/u.kpc**3).to(u.Msun/u.Mpc**3)

    # some Concentration models (e.g. bullock01, bhattacharya13) don't support array z internally, so this still needs to loop over halos
    def c(self, Concentration, MassDef):  # Concentration [], from self.z and self.logM
        cfunc = lambda z, M: concentration.concentration(M, z=z, model=Concentration, mdef=MassDef)
        return np.vectorize(cfunc)(self.z, 10**self.logM*self.cosmology.h) *u.dimensionless_unscaled

    # colossus's changeMassDefinition requires a scalar redshift, so unlike the other functions this still needs to loop over halos
    def logM_conv(self, newmdef, MassDef, Concentration, conc=None):  # Convert mass definition, from self.z and self.logM
        Mconvattr = mass_defs.changeMassDefinition
        if conc is not None:
            Mconvfunc = lambda z, logM, conc: np.vectorize(Mconvattr)(10**logM*self.cosmology.h, conc(z, logM), z, MassDef, newmdef)
            return lambda conc: np.log10(Mconvfunc(self.z, self.logM, conc)[0])
        else:
            Mconvfunc = lambda z, logM: np.vectorize(Mconvattr)(10**logM*self.cosmology.h, self.c(Concentration, MassDef).value, z, MassDef, newmdef)
            return np.log10(Mconvfunc(self.z, self.logM)[0])
