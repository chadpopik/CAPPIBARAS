"""
Rewrite of HaloModels.pyccl_model with the inputs flipped: z, logM, k, and R are
fixed at construction (as self.z, self.logM, self.k, self.R) instead of being passed
into every function call, while the model choices (MassDef, MassFunc, HaloBias,
Concentration) are now passed into each function call instead of being fixed at
construction.

All unit handling (stripping astropy units) happens once in __init__, so the functions
themselves just use self.z/self.logM/self.k/self.R directly with no further unit handling.

z/logM/k/R are also reshaped onto separate broadcasting axes in __init__, so that any
function combining two or more of them (e.g. bh(z, M), Plin(k, z)) naturally comes out
shaped (n_r_or_k, n_z, n_M) rather than requiring the caller to line up matching lengths.
Any of z/logM/k/R may be left as None; __init__ never fails because one is missing, only
whichever function actually needs it does.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import numpy as np
import astropy.units as u


def unitinput(val, unit=u.dimensionless_unscaled):
    return (val if isinstance(val, u.Quantity) else val*unit).to(unit).value


class pyccl_model:  # https://ccl.readthedocs.io/en/latest/index.html
    def __init__(self, z=None, logM=None, k=None, R=None, **cosmo_params):
        import pyccl as ccl
        self.ccl = ccl

        defaultparams = ccl.CosmologyVanillaLCDM().to_dict()
        self.cosmo = ccl.Cosmology(**(defaultparams | self.fixnames(cosmo_params)))

        z = None if z is None else unitinput(z)
        self.z = None if z is None else np.array(z, ndmin=1)[:, None]  # z axis (middle)

        logM = logM.to(u.dimensionless_unscaled).value if isinstance(logM, u.Quantity) else logM
        self.logM = None if logM is None else np.array(logM, ndmin=1)  # M axis (last)

        k = None if k is None else unitinput(k, 1/u.Mpc)
        self.k = None if k is None else np.array(k, ndmin=1)[:, None, None]  # k axis (first)

        R = None if R is None else unitinput(R, u.Mpc)
        self.R = None if R is None else np.array(R, ndmin=1)[:, None, None]  # r axis (first)

    def fixnames(self, inputs):  # Format names properly
        if 'Ob0' in inputs and 'Omega_b' not in inputs: inputs['Omega_b'] = inputs['Ob0']
        if 'Om0' in inputs and 'Omega_m' not in inputs: inputs['Omega_m'] = inputs['Om0']
        if 'H0' in inputs and 'h' not in inputs: inputs['h'] = inputs['H0']/100
        if 'Tcmb0' in inputs and 'T_CMB' not in inputs: inputs['T_CMB'] = inputs['TCMB0']
        if 'Om0' in inputs and 'Ob0' in inputs and 'Omega_c' not in inputs: inputs['Omega_c'] = inputs['Om0'] - inputs['Ob0']
        return inputs

    def H(self):  # Hubble Function [km/s/Mpc], from self.z
        rawfunc = lambda z: self.cosmo.h_over_h0(a=1/(1+z))
        return np.vectorize(rawfunc)(self.z) * self.cosmo._params['H0'] *u.km/u.s/u.Mpc

    def chi(self):  # Comoving Distance [Mpc], from self.z
        rawfunc = lambda z: self.cosmo.comoving_radial_distance(a=1/(1+z))
        return np.vectorize(rawfunc)(self.z) *u.Mpc

    def dA(self):  # Angular Diameter Distance [Mpc], from self.z
        rawfunc = lambda z: self.cosmo.angular_diameter_distance(a1=1, a2=1/(1+z))
        return np.vectorize(rawfunc)(self.z) *u.Mpc

    def rhoc(self):  # Critical Density [Msol/Mpc^3], from self.z
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='critical', is_comoving=False)
        return np.vectorize(rawfunc)(self.z) *u.Msun/u.Mpc**3

    def rhom(self):  # Mean Matter Density [Msol/Mpc^3], from self.z
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='matter', is_comoving=False)
        return np.vectorize(rawfunc)(self.z) *u.Msun/u.Mpc**3

    def Vcom(self):  # Comoving Volume [Mpc^3], from self.z
        rawfunc = lambda z: self.cosmo.comoving_volume(a=1/(1+z))
        return np.vectorize(rawfunc)(self.z) *u.Mpc**3

    # TODO: check on what function/power spec is calculated here, if it's what we want
    def Plin(self):  # Linear Matter Power Spectrum [Mpc^3], from self.k and self.z
        rawfunc = lambda k, z: self.cosmo.linear_matter_power(k, a=1/(1+z))
        return np.vectorize(rawfunc)(self.k, self.z) *u.Mpc**3

    # TODO: units?
    def NFW_r(self, MassDef, Concentration, trunc=False):  # NFW profile in real space, from self.R, self.z, self.logM
        NFWattr = self.ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=MassDef, concentration=Concentration, truncated=trunc)
        rawfunc = lambda r, z, M: NFWattr.real(cosmo=self.cosmo, r=r, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(self.R, self.z, 10**self.logM) *u.Msun/u.Mpc**3

    # TODO: units?
    def NFW_k(self, MassDef, Concentration, trunc=False):  # NFW profile in fourier space, from self.k, self.z, self.logM
        NFWattr = self.ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=MassDef, concentration=Concentration, truncated=trunc)
        rawfunc = lambda k, z, M: NFWattr.fourier(cosmo=self.cosmo, k=k, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(self.k, self.z, 10**self.logM) *u.Msun/u.Mpc**3

    def bh(self, HaloBias, MassDef):  # Halo Bias [], from self.z and self.logM
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        bhattr = self.ccl.halos.hbias.HaloBias.from_name(HaloBias)
        rawfunc = lambda z, M: bhattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(self.z, 10**self.logM) *u.dimensionless_unscaled

    def dndlogm(self, MassFunc, MassDef):  # Halo Mass Function [1/Mpc^3/dex], from self.z and self.logM
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        dndlogattr = self.ccl.halos.hmfunc.MassFunc.from_name(MassFunc)
        rawfunc = lambda z, M: dndlogattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(self.z, 10**self.logM) *1/u.Mpc**3/u.dex

    def c(self, Concentration, MassDef, c0=1):  # Concentration [], from self.z and self.logM
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        cattr = self.ccl.halos.concentration.Concentration.from_name(Concentration)
        if Concentration!='Constant':
            rawfunc = lambda z, M: cattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        else:
            rawfunc = lambda z, M: cattr(mass_def=mdefattr, c=c0)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(self.z, 10**self.logM) *u.dimensionless_unscaled

    def logM_conv(self, newmdef, MassDef, Concentration, c0=1):  # Convert mass definition, from self.z and self.logM
        mdefinattr = self.ccl.halos.MassDef.from_name(MassDef)
        mdefoutattr = self.ccl.halos.MassDef.from_name(newmdef)
        mconvattr = self.ccl.halos.massdef.mass_translator
        cattr = self.ccl.halos.concentration.Concentration.from_name(Concentration)
        cfunc = cattr(mass_def=mdefinattr) if Concentration!='Constant' else cattr(mass_def=mdefinattr, c=c0)
        mconvfunc = lambda z, M: mconvattr(mass_in=mdefinattr, mass_out=mdefoutattr, concentration=cfunc)(self.cosmo, M, a=1/(1+z))
        return np.log10(np.vectorize(mconvfunc)(self.z, 10**self.logM))
