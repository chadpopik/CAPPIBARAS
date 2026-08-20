"""
Rewrite of HaloModels.pyccl_model. z, logM, k, and r are passed into every function call
rather than being fixed at construction, and so are the model choices (MassDef, MassFunc,
HaloBias, Concentration) -- except that HaloModel.__init__ can also store defaults for
those model choices, which every function below falls back to when the corresponding
argument is left as None.

All unit handling (stripping astropy units) happens inside each function, so the caller
can pass z/logM/k/r as bare numbers/arrays or as astropy Quantities.

z/logM/k/r are also reshaped onto separate broadcasting axes inside each function, so that
any function combining two or more of them (e.g. bh(z, M), Plin(k, z)) naturally comes out
shaped (n_r_or_k, n_z, n_M) rather than requiring the caller to line up matching lengths.

The original class in Models/HaloModels.py is untouched; this is a standalone rewrite.
"""


import inspect
import numpy as np
import astropy.units as u
import pyccl as ccl


def unitinput(val, unit=u.dimensionless_unscaled):
    return (val if isinstance(val, u.Quantity) else val*unit).to(unit).value


def reshape_z(z):  # plain z, natural shape preserved (scalar stays scalar)
    return None if z is None else unitinput(z)


def reshape_z_axis(z):  # z axis (middle), for combining with logM/k/r
    z = reshape_z(z)
    return None if z is None else np.array(z, ndmin=1)[:, None]


def reshape_logM(logM):  # M axis (last)
    logM = logM.to(u.dimensionless_unscaled).value if isinstance(logM, u.Quantity) else logM
    return None if logM is None else np.array(logM, ndmin=1)


def reshape_k(k):  # k axis (first)
    return None if k is None else np.array(unitinput(k, 1/u.Mpc), ndmin=1)[:, None, None]


def reshape_r(r):  # r axis (first)
    return None if r is None else np.array(unitinput(r, u.Mpc), ndmin=1)[:, None, None]


class Cosmology:  # https://ccl.readthedocs.io/en/latest/index.html
    @staticmethod
    def valid_options():  # valid name strings CCL accepts for each cosmology model choice, discovered from CCL itself
        return {
            'TransFunc': list(ccl.cosmology.TransferFunctions._member_names_),  # Transfer Functions (cosmo: transfer_function=...)
            'PowSpec': list(ccl.cosmology.MatterPowerSpectra._member_names_),  # Power Spectrum calc (cosmo: matter_power_spectrum=...)
        }

    @classmethod
    def _check_model(cls, kind, name):  # validate a model name; on a bad one, raise with the list of valid options
        if name is None:
            return name
        opts = cls.valid_options()[kind]
        match = next((o for o in opts if o.lower() == str(name).lower()), None)
        if match is None and kind == 'MassDef' and str(name)[-1:].lower() in ('c', 'm') and str(name)[:-1].isdigit():
            match = name  # arbitrary overdensity, e.g. "300c" / "500m"
        if match is None:
            extra = ' (or any "<Delta>c"/"<Delta>m", e.g. "300c")' if kind == 'MassDef' else ''
            raise ValueError(f"Invalid {kind} {name!r}. Valid options: {', '.join(opts)}{extra}.")
        # CCL's cosmology args want the lowercase form; its halo `from_name` registries are case-sensitive on the canonical name
        return match.lower() if kind in ('TransFunc', 'PowSpec') else match

    def __init__(self, **cosmo_params):
        self.ccl = ccl

        for arg, kind in [('transfer_function', 'TransFunc'), ('matter_power_spectrum', 'PowSpec')]:
            if arg in cosmo_params:
                cosmo_params[arg] = self._check_model(kind, cosmo_params[arg])

        defaultparams = ccl.CosmologyVanillaLCDM().to_dict()

        if 'Omega_m' in cosmo_params:  # CCL wants Omega_c directly; derive it from Omega_m - Omega_b
            if 'Omega_c' in cosmo_params:
                raise ValueError("Specify only one of Omega_m or Omega_c, not both")
            Omega_m = cosmo_params.pop('Omega_m')
            cosmo_params['Omega_c'] = Omega_m - cosmo_params.get('Omega_b', defaultparams['Omega_b'])

        inputdict = {'extra_parameters':{}}
        for k, v in cosmo_params.items():
            if k in defaultparams: inputdict[k] = v
            else: 
                inputdict['extra_parameters'][k]=v
                if k!='name':
                    print(f"Warning: {k} = {v} not incorporated into cosmology")
        self.cosmo = ccl.Cosmology(**(defaultparams | inputdict))
        self.cosmoparams = self.cosmo.to_dict()
        for k, v in self.cosmo.to_dict().items():
            setattr(self, k, v)  # store all cosmology parameters as attributes for easy access
            if k=='extra_parameters':
                for k2, v2 in v.items():
                    setattr(self, k2, v2)  # store extra parameters as attributes too

        self.H0 = self.h*100*u.km/u.s/u.Mpc
        self.Omega_m = self.Omega_c + self.Omega_b

    def H(self, z):  # Hubble Function [km/s/Mpc]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.h_over_h0(a=1/(1+z))
        return np.vectorize(rawfunc)(z) * self.cosmo._params['H0'] *u.km/u.s/u.Mpc

    def chi(self, z):  # Comoving Distance [Mpc]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.comoving_radial_distance(a=1/(1+z))
        return np.vectorize(rawfunc)(z) *u.Mpc

    def dA(self, z):  # Angular Diameter Distance [Mpc]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.angular_diameter_distance(a1=1, a2=1/(1+z))
        return np.vectorize(rawfunc)(z) *u.Mpc

    def rhoc(self, z):  # Critical Density [Msol/Mpc^3]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='critical', is_comoving=False)
        return np.vectorize(rawfunc)(z) *u.Msun/u.Mpc**3

    def rhom(self, z):  # Mean Matter Density [Msol/Mpc^3]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='matter', is_comoving=False)
        return np.vectorize(rawfunc)(z) *u.Msun/u.Mpc**3

    def Vcom(self, z):  # Comoving Volume [Mpc^3]
        z = reshape_z(z)
        rawfunc = lambda z: self.cosmo.comoving_volume(a=1/(1+z))
        return np.vectorize(rawfunc)(z) *u.Mpc**3
    
    # TODO: check on what function/power spec is calculated here, if it's what we want
    def Plin(self, k, z):  # Linear Matter Power Spectrum [Mpc^3]
        k = reshape_k(k)
        z = reshape_z_axis(z)
        rawfunc = lambda k, z: self.cosmo.linear_matter_power(k, a=1/(1+z))
        return np.vectorize(rawfunc)(k, z) *u.Mpc**3


class HaloModel(Cosmology):
    @staticmethod
    def valid_options():  # cosmology options plus the halo-model choices, discovered from CCL itself
        return {**Cosmology.valid_options(),
            'MassDef': [m.replace('MassDef', '') for m in dir(ccl.halos) if 'MassDef' in m and m != 'MassDef'],  # Mass Definitions (also any "<Delta>c"/"<Delta>m")
            'MassFunc': [m.replace('MassFunc', '') for m in dir(ccl.halos.hmfunc) if 'MassFunc' in m and m != 'MassFunc'],  # Halo Mass Function Models
            'HaloBias': [m.replace('HaloBias', '') for m in dir(ccl.halos.hbias) if 'HaloBias' in m and m != 'HaloBias'],  # Halo Bias Models
            'Concentration': [m.replace('Concentration', '') for m in dir(ccl.halos.concentration) if 'Concentration' in m and m != 'Concentration'],  # Concentration Models
        }

    def __init__(self, MassDef=None, MassFunc=None, HaloBias=None, Concentration=None, **cosmo_params):
        super().__init__(**cosmo_params)

        # model choices set here become the default for every method below, but can still be overridden per call.
        # Each is validated now so a bad name fails at construction (listing the valid options) rather than
        # deep inside a later method call.
        self.MassDef = self._check_model('MassDef', MassDef)
        self.MassFunc = self._check_model('MassFunc', MassFunc)
        self.HaloBias = self._check_model('HaloBias', HaloBias)
        self.Concentration = self._check_model('Concentration', Concentration)

    # TODO: units?
    def NFW_r(self, r, z, logM, MassDef=None, Concentration=None, trunc=False):  # NFW profile in real space
        MassDef = self.MassDef if MassDef is None else MassDef
        Concentration = self.Concentration if Concentration is None else Concentration
        r = reshape_r(r)
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        NFWattr = self.ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=MassDef, concentration=Concentration, truncated=trunc)
        rawfunc = lambda r, z, M: NFWattr.real(cosmo=self.cosmo, r=r, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(r, z, 10**logM) *u.Msun/u.Mpc**3

    # TODO: units?
    def NFW_k(self, k, z, logM, MassDef=None, Concentration=None, trunc=True, normalized=True):  # NFW profile in fourier space
        MassDef = self.MassDef if MassDef is None else MassDef
        Concentration = self.Concentration if Concentration is None else Concentration
        k = reshape_k(k)
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        NFWattr = self.ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=MassDef, concentration=Concentration, truncated=trunc)
        rawfunc = lambda k, z, M: NFWattr.fourier(cosmo=self.cosmo, k=k, M=M, a=1/(1+z))
        prof = np.vectorize(rawfunc)(k, z, 10**logM)
        if normalized:  # untruncated NFW has infinite mass and no k->0 limit; truncated -> M, so /M gives a profile that -> 1 as k->0
            prof = prof / 10**logM  # logM is on the last axis; broadcasts over the M axis
        return prof *u.dimensionless_unscaled

    def bh(self, z, logM, HaloBias=None, MassDef=None):  # Halo Bias []
        HaloBias = self.HaloBias if HaloBias is None else HaloBias
        MassDef = self.MassDef if MassDef is None else MassDef
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        bhattr = self.ccl.halos.hbias.HaloBias.from_name(HaloBias)
        rawfunc = lambda z, M: bhattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(z, 10**logM) *u.dimensionless_unscaled

    def dndlogm(self, z, logM, MassFunc=None, MassDef=None):  # Halo Mass Function [1/Mpc^3/dex]
        MassFunc = self.MassFunc if MassFunc is None else MassFunc
        MassDef = self.MassDef if MassDef is None else MassDef
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        dndlogattr = self.ccl.halos.hmfunc.MassFunc.from_name(MassFunc)
        rawfunc = lambda z, M: dndlogattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(z, 10**logM) *1/u.Mpc**3/u.dex

    def c(self, z, logM, Concentration=None, MassDef=None, c0=1):  # Concentration []
        Concentration = self.Concentration if Concentration is None else Concentration
        MassDef = self.MassDef if MassDef is None else MassDef
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        mdefattr = self.ccl.halos.MassDef.from_name(MassDef)
        cattr = self.ccl.halos.concentration.Concentration.from_name(Concentration)
        if Concentration!='Constant':
            rawfunc = lambda z, M: cattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        else:
            rawfunc = lambda z, M: cattr(mass_def=mdefattr, c=c0)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(z, 10**logM) *u.dimensionless_unscaled

    def logM_conv(self, z, logM, newmdef, MassDef=None, Concentration=None, c0=1):  # Convert mass definition
        MassDef = self.MassDef if MassDef is None else MassDef
        Concentration = self.Concentration if Concentration is None else Concentration
        newmdef = self._check_model('MassDef', newmdef)
        z = reshape_z_axis(z)
        logM = reshape_logM(logM)
        mdefinattr = self.ccl.halos.MassDef.from_name(MassDef)
        mdefoutattr = self.ccl.halos.MassDef.from_name(newmdef)
        mconvattr = self.ccl.halos.massdef.mass_translator
        cattr = self.ccl.halos.concentration.Concentration.from_name(Concentration)
        cfunc = cattr(mass_def=mdefinattr) if Concentration!='Constant' else cattr(mass_def=mdefinattr, c=c0)
        mconvfunc = lambda z, M: mconvattr(mass_in=mdefinattr, mass_out=mdefoutattr, concentration=cfunc)(self.cosmo, M, a=1/(1+z))
        return np.log10(np.vectorize(mconvfunc)(z, 10**logM))
