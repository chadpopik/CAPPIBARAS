"""
Collection of Halo models functions for use in cosmology.
"""


import numpy as np
import astropy
import astropy.cosmology
import astropy.units as u
import astropy.constants as c



def y_to_uK(nu, T_CMB, **kwargs):  # Here to convert measurements
    x = (c.h * nu / (c.k_B * T_CMB)).decompose().value
    fnu = x / np.tanh(x / 2.0) - 4.0
    return fnu*T_CMB.to(u.uK)

class BaseModel:
    def check_inputs(self, inputs, reqlist):  # check all required inputs are present and valid
        for req in reqlist:  # for every input required (model/data subset/etc)
            if req not in inputs.keys():  # if the required thing isn't in the input 
                raise ValueError(f"Missing {req} is required")  # print an error
            if self.required[req] is not None:  # if there's a list of options for that input type (it's a str to decide parameters)
                if inputs[req] not in self.required[req]:  # If that input isn't in the list
                    raise NameError(f"{req} {inputs[req]} doesn't exist, choose from availables: {self.required[req]}")  # print an error
            setattr(self, req, inputs[req])  # set that input as a attribute if it's there and valid
            
    def r200c(self, zs, logMs):
        return ((10**logMs*u.Msun/(4/3*np.pi*200*self.rhoc(zs)))**(1/3)).to(u.Mpc)  # Mpc
    
    def volumes(self, zs, **kwargs):
        dz = (zs[1]-zs[0])  # size of Redshift slices
        return (self.Vcom(zs+dz/2)-self.Vcom(zs-dz/2))/(1+zs)**3  # Calculate non-comoving shell for every z



class astropy_model(BaseModel):  # https://docs.astropy.org/en/stable/cosmology/index.html
    required = {'Parameters':['H0', 'Om0', 'Ob0', 'Tcmb0', 'Neff', 'm_nu']}  # TODO: add more parameter options?
    
    def __init__(self, **kwargs):
        Planck = astropy.cosmology.Planck18
        
        cosmopars = {key: val for key, val in kwargs.items() if key in self.required.parameters}
        self.cosmology = Planck.clone(**(Planck.parameters | cosmopars))
    
    def H(self, zs):
        return self.cosmology.H(zs)  # km/s/Mpc
        
    def chi(self, zs):
        return self.cosmology.comoving_distance(zs)  # Mpc
    
    def dA(self, zs):
        return self.cosmology.angular_diameter_distance(zs)  # Mpc
    
    def rhoc(self, zs):
        return self.cosmology.critical_density(zs).to(u.Msun/u.Mpc**3)  # Msun/Mpc^3
        
    def Vcom(self, zs, **kwargs):
        return self.cosmology.comoving_volume(zs)  # Mpc^3


class pyccl_model(BaseModel):  # https://ccl.readthedocs.io/en/latest/index.html
    import pyccl as ccl  # Import package only if using this model
    ccl = ccl

    required={
        'MassDef':[mod.replace('MassDef','') for mod in dir(ccl.halos) if 'MassDef' in mod and mod!='MassDef'],
        'MassFunc' : [mod.replace('MassFunc','') for mod in dir(ccl.halos.hmfunc) if 'MassFunc' in mod and mod!='MassFunc'],  # can actually use any overdensity number for c and m?
        'HaloBias' : [mod .replace('HaloBias','') for mod in dir(ccl.halos.hbias) if 'HaloBias' in mod and mod!='HaloBias'],
        'Concentration' : [mod.replace('Concentration','')  for mod in dir(ccl.halos.concentration) if 'Concentration' in mod and mod!='Concentration'],
        'Parameters': ['h', 'Omega_c', 'Omega_b', 'T_CMB', 'sigma8', 'n_s', 'Neff'],  # TODO: more parameter options?
    }    

    def __init__(self, inputs, **kwargs):
        self.check_inputs(inputs, ['MassDef', 'MassFunc', 'HaloBias', 'Concentration'])

        defaultparams = self.ccl.CosmologyParams.get_params_dict(self.ccl.cosmology.CosmologyVanillaLCDM()._params)  # not all parameters will work in the cosmology
        defaultparams = {key: val for key, val in defaultparams.items() if key in self.required['Parameters']}
        inputpars = {key: val for key, val in kwargs.items() if key in self.required['Parameters']}
        self.cosmology = self.ccl.Cosmology(**(defaultparams | inputpars))
        
    def H(self, zs):
        Hfunc = lambda z: np.vectorize(self.cosmology.h_over_h0)(a=1/(1+zs))
        return Hfunc(zs) * self.cosmology._params['H0'] *u.km/u.s/u.Mpc
    
    def chi(self, zs):
        chifunc = lambda z: np.vectorize(self.cosmology.comoving_radial_distance)(a=1/(1+z))
        return chifunc(zs) *u.Mpc
        
    def dA(self, zs):
        dAfunc = lambda z: np.vectorize(self.cosmology.angular_diameter_distance)(a1=1/(1+z))
        return dAfunc(zs) *u.Mpc
        
    def rhoc(self, zs):
        rhocfunc = lambda z: np.vectorize(self.cosmology.rho_x)(a=1/(1+z), species='critical', is_comoving=False)
        return rhocfunc(zs) *u.Msun/u.Mpc**3
    
    def Vcom(self, zs, **kwargs):
        Vcomfunc = lambda z: np.vectorize(self.cosmology.comoving_volume)(a=1/(1+z))
        return Vcomfunc(zs) *u.Mpc**3

    def bh(self, zs, logMs):
        bhattr = getattr(self.ccl.halos.hbias, f"HaloBias{self.HaloBias}")(mass_def=self.MassDef)
        bhfunc = lambda z, logM: np.vectorize(bhattr)(cosmo=self.cosmology, M=10**logM, a=1/(1+z))
        return bhfunc(zs, logMs) *u.dimensionless_unscaled

    def dndlogm(self, zs, logMs):
        dndlogattr = getattr(self.ccl.halos.hmfunc, f"MassFunc{self.MassFunc}")(mass_def=self.MassDef)
        dndlogfunc = lambda z, logM: np.vectorize(dndlogattr)(cosmo=self.cosmology, M=10**logM, a=1/(1+z))
        return dndlogfunc(zs, logMs) *1/u.Mpc**3/u.dex
    
    def c(self, zs, logMs):
        cattr = getattr(self.ccl.halos.concentration, f"Concentration{self.Concentration}")(mass_def=self.MassDef)
        cfunc = lambda z, logM: np.vectorize(cattr)(cosmo=self.cosmology, M=10**logM, a=1/(1+z))
        return cfunc(zs, logMs) *u.dimensionless_unscaled
        
    def Plin(self, ks, zs):
        Plinfunc = lambda k, z: np.vectorize(self.cosmology.linear_matter_power)(k, a=1/(1+z))
        return Plinfunc(ks, zs) *u.Mpc**3
        
    # Mass conversion
    # TODO 2: Check mass conversions for proper functioning
    def Mconv(self, logmshalo, zs, mdefin, mdefout, **kwargs):
        massconv = self.ccl.halos.massdef.mass_translator(mass_in=mdefin, mass_out=mdefout, concentration=self.concmod)
        return np.array([np.log10(massconv(self.cosmo, 10**logmshalo, 1/(1+z))) for z in np.atleast_1d(zs).flatten()])


class colossus_model(BaseModel):  # https://ccl.readthedocs.io/en/latest/index.html
    import colossus
    from colossus.cosmology import cosmology, power_spectrum
    from colossus.halo import concentration, mass_defs, mass_adv
    from colossus.lss import bias, mass_function
    colossus = colossus
    
    required={
        'Parameters':['H0', 'Om0', 'Ob0', 'Tcmb0', 'Neff', 'ns', 'sigma8'],  # TODO: more parameter options?
        'MassFunc': list(colossus.lss.mass_function.models.keys()),
        'HaloBias': list(colossus.lss.bias.models.keys()),
        'Concentration': list(colossus.halo.concentration.models.keys()),
        'PlinMod': list(colossus.cosmology.cosmology.power_spectrum.models.keys()),
        'MassDef': ['200c', '500c', '200m', 'fof', 'vir'],  # TODO: get actual list for this
              } 
    
    def __init__(self, inputs, **kwargs):
        self.check_inputs(inputs, ['MassDef', 'MassFunc', 'HaloBias', 'Concentration', 'PlinMod'])
        
        inputpars = {key: val for key, val in kwargs.items() if key in self.required['Parameters']}
        self.cosmology = self.colossus.cosmology.cosmology.setCosmology('planck18', inputpars)
        
    def H(self, zs):
        Hfunc = lambda z: np.vectorize(self.cosmology.Hz)(z=z)
        return Hfunc(zs) *u.km/u.s/u.Mpc
    
    def chi(self, zs):
        chifunc = lambda z: np.vectorize(self.cosmology.comovingDistance)(z_min=0, z_max=z)
        return chifunc(zs) *u.Mpc/self.cosmology.h
    
    def dA(self, zs):
        dAfunc = lambda z: np.vectorize(self.cosmology.angularDiameterDistance)(z=z)
        return dAfunc(zs) *u.Mpc/self.cosmology.h
    
    def rhoc(self, zs):
        rhocfunc = lambda z: np.vectorize(self.cosmology.rho_c)(z=z)
        return (rhocfunc(zs) *u.Msun/u.kpc**3*self.cosmology.h**2).to(u.Msun/u.Mpc**3)
                
    def c(self, zs, logMs):
        cattr = self.colossus.halo.concentration.concentration
        cfunc = lambda z, logM: np.vectorize(cattr)(10**logM, z=z, model=self.Concentration, mdef=self.MassDef)
        return cfunc(zs, logMs) *u.dimensionless_unscaled
    
    def dndlogm(self, zs, logMs):
        dndlogmattr = self.colossus.lss.mass_function.massFunction
        dndlogmfunc = lambda z, logM: np.vectorize(dndlogmattr)(10**logM, z=z, model=self.MassFunc, mdef=self.MassDef, q_out='dndlnM')
        return dndlogmfunc(zs, logMs) *1/u.Mpc**3/u.dex
    
    def bh(self, zs, logMs):
        bhattr = self.colossus.lss.bias.haloBias
        bhfunc = lambda z, logM: np.vectorize(bhattr)(10**logM, z=z, model=self.HaloBias, mdef=self.MassDef)
        return bhfunc(zs, logMs) *u.dimensionless_unscaled
    
    def Plin(self, ks, zs):
        Plinfunc = lambda k, z: self.cosmology.matterPowerSpectrum(k=k, z=z, model=self.PlinMod)
        return Plinfunc(ks, zs) *u.Mpc**3





# class hmf_model(BaseModel):  # https://hmf.readthaedocs.io/en/latest/index.html
#     mfuncs = ['Angulo', 'AnguloBound', 'Behroozi', 'Bhattacharya', 'Bocquet200cDMOnly', 'Bocquet200cHydro', 'Bocquet200mDMOnly', 'Bocquet200mHydro', 'Bocquet500cDMOnly', 'Bocquet500cHydro', 'Courtin', 'Crocce', 'FittingFunction', 'Ishiyama', 'Jenkins', 'Manera', 'PS', 'Peacock', 'Pillepich', 'Reed03', 'Reed07', 'SMT', 'ST', 'SimDetails', 'Tinker08', 'Tinker10', 'Union', 'Warren', 'Watson', 'Watson_FoF']
#     mdefs = ['FOF','SOCritical','SOGeneric','SOMean','SOVirial','SphericalOverdensity']
#     def __init__(self, spefs):
#         import hmf  # Import package only if using this model
#         self.hmf = hmf
        
#         self.checkspefs(spefs, required=['mdef', 'mfunc'])

#     # Halo Mass Function
#     def HMF(self, zs, logmshalo, hh=0.7, Omega_b=0.044, Omega_m=0.3, Omega_L=0.7, T_CMB=2.275, **kwargs):
#         # TODO 1: Check about adding more detail to the cosmology setups
#         cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)
#         logmshalo = np.atleast_1d(logmshalo)+np.log10(hh)
#         dlog10m = logmshalo[1]-logmshalo[0]

#         # Function only takes one z at a time so use list comprehension and then combine
#         if zs.ndim==2: zs=zs[:, 0]
#         haloMFsraw = [self.hmf.MassFunction(z=z, Mmin=logmshalo.min(), Mmax=logmshalo.max()+dlog10m, dlog10m=dlog10m, hmf_model=self.mfunc, mdef_model=self.mdef, cosmo_model=cosmo) for z in np.atleast_1d(zs).flatten()]
#         HMF_m_z = np.array([np.interp(logmshalo, np.log10(haloMFraw.m), haloMFraw.dndlog10m)*hh**3 for haloMFraw in haloMFsraw])
#         return HMF_m_z




# class hmvec(BASEHMF):
#     mfuncs = ['tinker', 'sheth-torman']
#     mdefs = ['vir', 'mean']
#     def __init__(self, mfunc, mdef):
#         sys.path.append("/global/homes/c/cpopik/")
#         import hmvec.hmvec.hmvec as hm
#         self.hm = hm
        
#         self.checkmodel(self.mfuncs, mfunc)
#         self.mfunc = mfunc
#         self.checkmodel(self.mdefs, mdef)
#         self.mdef = mdef
        
        
#     def HMF(self, ms, zs):
#         self.hmvecmodel = self.hm.HaloModel(ms = ms, zs = zs, ks=np.linspace(1e-5, 200, 1),
#             mass_function=self.mfunc,mdef=self.mdef,
#             nfw_numeric=False, skip_nfw=False, accurate_sigma2=False)
#         return self.hmvecmodel.nzm