"""
Collection of Halo models functions for use in cosmology.
"""


import numpy as np
import astropy
import astropy.cosmology
import astropy.units as u
import astropy.constants as c
from scipy.integrate import quad



def y_to_uK(nu, T_CMB, **kwargs):  # Here to convert measurements
    x = (c.h * nu / (c.k_B * T_CMB)).decompose().value
    fnu = x / np.tanh(x / 2.0) - 4.0
    return fnu*T_CMB.to(u.uK)


class BaseModel:
    def check_inputs(self, inpdict={}, optdict={}):  # dict of inputs, dict of (lists of) options for each input
        for optkey, optvals in optdict.items():  # for all options that have lists
            if optkey in inpdict and inpdict[optkey] not in optvals:  # if the input contains one BUT it's not in the list
                if isinstance(inpdict[optkey], str) and inpdict[optkey].lower() in [v.lower() for v in optvals]: pass
                else: raise NameError(f"{optkey} {inpdict[optkey]} doesn't exist, choose from: {optvals}")  # print an error

    def require_input(self, inpdict, reqlist):  # dict of given inputs, list of required inputs
        for req in reqlist:  # for all required inputs
            if req not in inpdict:  # if input doesn't have it
                raise ValueError(f"Missing {req} is required" )  # print an error

    def setup(self, inputs):
        self.check_inputs(inputs, self.options)  # check inputs
        self.require = lambda reqlist: self.require_input(inputs, reqlist)  # make requirement function
        for inpkey, inpval in inputs.items(): setattr(self, inpkey, inpval)  # set all inputs as attributes
        self.params = {inpkey:inpval for inpkey, inpval in inputs.items() if inpkey in self.Parameters}  # collect the ones that are input parameters
        for pkey, pval in self.params.items():
            try: self.params[pkey] = pval.value
            except: pass

    def rdel(self, mdef):
        if mdef[-1]=='c': rho = lambda zs: int(mdef[:-1])*self.rhoc(zs)
        elif mdef[-1]=='m': rho = lambda zs: int(mdef[:-1])*self.rhom(zs)
        return lambda zs, logMs: ((10**logMs*u.Msun/(4/3*np.pi*rho(zs)))**(1/3)).to(u.Mpc)  # Mpc

    def volumes(self, zs, **kwargs):
        dz = (zs[1]-zs[0])  # size of Redshift slices
        return (self.Vcom(zs+dz/2)-self.Vcom(zs-dz/2))/(1+zs)**3  # Calculate non-comoving shell for every z



class astropy_model(BaseModel):  # https://docs.astropy.org/en/stable/cosmology/index.htm
    Parameters = list(astropy.cosmology.Planck18.parameters.keys())  # Available cosmological parameter inputs
    options = {
        'Cosmology': list(astropy.cosmology.available),  # Available Cosmologies
        'MassDef': ['virial', '200c', '200m',  '500c', '500m'],  # Avaiable Mass Definitions
    }

    def __init__(self, inputdict={}, **inputvars):        
        self.setup(inputdict | inputvars)
        self.Cosmology = self.Cosmology.capitalize() if hasattr(self, 'Cosmology') else 'Planck18'  # Format Properly, Set Planck18 as default

        cosmo = getattr(astropy.cosmology, self.Cosmology)  # Get Base Cosmology
        self.cosmology = cosmo.clone(**(cosmo.parameters | self.params))  # Update parameters
        self.params = {(k[1:] if k.startswith('_') else k):v for k, v in self.cosmology.__dict__.items()}  # Create dictionary of params

    def H(self, zs):  # Hubble function [km/s/Mpc], from redshift []
        return self.cosmology.H(zs)

    def chi(self, zs):  # comoving distance [Mpc], from redshift []
        return self.cosmology.comoving_distance(zs)

    def dA(self, zs):  # angular diameter distance [Mpc], from redshift []
        return self.cosmology.angular_diameter_distance(zs)

    def rhoc(self, zs):  # critical density [Msun/Mpc^3], from redshift []
        return self.cosmology.critical_density(zs).to(u.Msun/u.Mpc**3)
    
    def rhom(self, zs):  # mean matter density [Msun/Mpc^3], from redshift []
        return self.rhoc(zs) * self.params['Om0']

    def Vcom(self, zs, **kwargs):  # comoving volume [Mpc^3], from redshift []
        return self.cosmology.comoving_volume(zs)

    # def rdel(self, zs, logMs):
    #     NFWmodel = lambda z, logM: np.vectorize(astropy.modeling.physical_models.NFW)(mass=10**logM, concentration=1, redshift=z, massfactor=self.MassDef, cosmo=self.cosmology)
    #     return np.array([nfw.r_virial for nfw in NFWmodel(zs, logMs)])  # Mpc
        



class pyccl_model(BaseModel):  # https://ccl.readthedocs.io/en/latest/index.html
    import pyccl as ccl  # Import package only if using this model
    ccl = ccl  # Set the package as an attribute of the class to use later
    Parameters = list(ccl.CosmologyVanillaLCDM().to_dict().keys())  # All cosmological parameters that can be input into the initialization
    options = {
        'TransFunc': ccl.cosmology.TransferFunctions._member_names_,  # Transfer Functions
        'PowSpec': ccl.cosmology.MatterPowerSpectra._member_names_,  # Power Spectrum calculations
        'MassDef': [mod.replace('MassDef','') for mod in dir(ccl.halos) if 'MassDef' in mod and mod!='MassDef'],  # Mass Definitions (TODO: can actually use any overdensity number for c and m?)
        'MassFunc' : [mod.replace('MassFunc','') for mod in dir(ccl.halos.hmfunc) if 'MassFunc' in mod and mod!='MassFunc'],  # Halo Mass Function Models
        'HaloBias' : [mod .replace('HaloBias','') for mod in dir(ccl.halos.hbias) if 'HaloBias' in mod and mod!='HaloBias'],  # Halo Bias Models
        'Concentration' : [mod.replace('Concentration','')  for mod in dir(ccl.halos.concentration) if 'Concentration' in mod and mod!='Concentration'],  # Concentration Models
    }    

    def __init__(self, inputdict={}, **inputvars):
        self.setup(self.fixnames(inputdict | inputvars))
        for opt in ['TransFunc', 'PowSpec']: 
            try: setattr(self, opt, getattr(self, opt).upper())
            except: pass
        for opt in ['MassDef', 'MassFunc', 'HaloBias', 'Concentration']: 
            try: setattr(self, opt, getattr(self, opt).capitalize())
            except: pass

        defaultparams = self.ccl.CosmologyVanillaLCDM().to_dict()
        self.cosmo = self.ccl.Cosmology(**(defaultparams | self.params))   
        self.params = self.cosmo._params.get_params_dict(self.cosmo._params)  # set dictionary of params

    def fixnames(self, inputs):  # Format names properly
        if 'Ob0' in inputs and 'Omega_b' not in inputs: inputs['Omega_b'] = inputs['Ob0']
        if 'Om0' in inputs and 'Omega_m' not in inputs: inputs['Omega_m'] = inputs['Om0']
        if 'H0' in inputs and 'h' not in inputs: inputs['h'] = inputs['H0']/100
        if 'Tcmb0' in inputs and 'T_CMB' not in inputs: inputs['T_CMB'] = inputs['TCMB0']
        if 'Om0' in inputs and 'Ob0' in inputs and 'Omega_c' not in inputs: inputs['Omega_c'] = inputs['Om0'] - inputs['Omega_b']
        return inputs


    def H(self, zs):  # Hubble Function [km/s/Mpc], from Redshift []
        rawfunc = lambda z: self.cosmo.h_over_h0(a=1/(1+z))
        return np.vectorize(rawfunc)(zs) * self.cosmo._params['H0'] *u.km/u.s/u.Mpc

    def chi(self, zs):  # Comoving Distance [Mpc], from Redshift []
        rawfunc = lambda z: self.cosmo.comoving_radial_distance(a=1/(1+z))
        return np.vectorize(rawfunc)(zs) *u.Mpc

    def dA(self, zs):  # Comoving Distance [Mpc], from Redshift []
        rawfunc = lambda z: self.cosmo.angular_diameter_distance(a1=1, a2=1/(1+z))
        return np.vectorize(rawfunc)(zs) *u.Mpc

    def rhoc(self, zs):  # Critical Density [Msol/Mpc^3], from Redshift []
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='critical', is_comoving=False)
        return np.vectorize(rawfunc)(zs) *u.Msun/u.Mpc**3

    def rhom(self, zs):  # Mean Matter Density [Msol/Mpc^3], from Redshift []
        rawfunc = lambda z: self.cosmo.rho_x(a=1/(1+z), species='matter', is_comoving=False)
        return np.vectorize(rawfunc)(zs) *u.Msun/u.Mpc**3

    def Vcom(self, zs, **kwargs):  # Comoving Volume, from Redshift []
        rawfunc = lambda z: self.cosmo.comoving_volume(a=1/(1+z))
        return np.vectorize(rawfunc)(zs) *u.Mpc**3
    
    # TODO: check on what function/power spec is calculated here, if it's what we want
    def Plin(self, ks, zs):  # Linear Matter Power Spectrum [Mpc^3], from Wavenumber [Mpc^-1] and Redshift []
        self.require(['PowSpec'])
        rawfunc = lambda k, z: self.cosmo.linear_matter_power(k, a=1/(1+z))
        return np.vectorize(rawfunc)(ks, zs) *u.Mpc**3
    
    def bh(self, zs, logMs):  # Halo Bias [], from Redshift [] and Halo Mass [Msun]
        self.require(['MassDef', 'HaloBias'])
        mdefattr = self.ccl.halos.MassDef.from_name(self.MassDef)
        bhattr = self.ccl.halos.hbias.HaloBias.from_name(self.HaloBias)
        rawfunc = lambda z, M: bhattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(zs, 10**logMs) *u.dimensionless_unscaled
    
    def dndlogm(self, zs, logMs):  # Halo Mass Function [1/Mpc^3/dex], from Redshift [] and Halo Mass [Msun]
        self.require(['MassDef', 'MassFunc'])
        mdefattr = self.ccl.halos.MassDef.from_name(self.MassDef)
        dndlogattr = self.ccl.halos.hmfunc.MassFunc.from_name(self.MassFunc)
        rawfunc = lambda z, M: dndlogattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
        return np.vectorize(rawfunc)(zs, 10**logMs) *1/u.Mpc**3/u.dex
    
    def c(self, zs, logMs, c0=1):  # Concentration [], from Redshift [] and Halo Mass [Msun]
        self.require(['MassDef', 'Concentration'])
        mdefattr = self.ccl.halos.MassDef.from_name(self.MassDef)
        cattr = self.ccl.halos.concentration.Concentration.from_name(self.Concentration)
        if self.Concentration!='Constant':
            rawfunc = lambda z, M: cattr(mass_def=mdefattr)(cosmo=self.cosmo, M=M, a=1/(1+z))
            return np.vectorize(rawfunc)(zs, 10**logMs) *u.dimensionless_unscaled
        else:
            rawfunc = lambda z, M: cattr(mass_def=mdefattr, c=c0)(cosmo=self.cosmo, M=M, a=1/(1+z))
            return np.vectorize(rawfunc)(zs, 10**logMs) *u.dimensionless_unscaled

    def logM_conv(self, zs, logMs, mdefout, c0=1):  # Convert mass definition
        self.require(['MassDef', 'Concentration'])
        self.check_inputs({'MassDef': mdefout}, self.options)
        mdefinattr = self.ccl.halos.MassDef.from_name(self.MassDef)
        mdefoutattr = self.ccl.halos.MassDef.from_name(mdefout)
        mconvattr = self.ccl.halos.massdef.mass_translator
        cattr = self.ccl.halos.concentration.Concentration.from_name(self.Concentration)
        cfunc = cattr(mass_def=mdefinattr) if self.Concentration!='Constant' else cattr(mass_def=mdefinattr, c=c0)
        mconvfunc = lambda z, M: mconvattr(mass_in=mdefinattr, mass_out=mdefoutattr, concentration=cfunc)(self.cosmo, M, a=1/(1+z))
        return np.log10(np.vectorize(mconvfunc)(zs, 10**logMs))



class colossus_model(BaseModel):  # https://ccl.readthedocs.io/en/latest/index.html
    import colossus
    from colossus.cosmology import cosmology, power_spectrum
    from colossus.halo import concentration, mass_defs, mass_adv
    from colossus.lss import bias, mass_function
    colossus = colossus
    
    Parameters = list({i for k in cosmology.cosmologies.values() for i in k.keys()})  # cosmological parameters
    options={
        'Cosmology': list(cosmology.cosmologies.keys()),
        'MassFunc': list(colossus.lss.mass_function.models.keys()),
        'HaloBias': list(colossus.lss.bias.models.keys()),
        'Concentration': list(colossus.halo.concentration.models.keys()),
        'PlinMod': list(colossus.cosmology.cosmology.power_spectrum.models.keys()),
        'MassDef': ['fof', 'vir', 'sp-apr-mn', '200c', '500c', '200m', '500m'],  # TODO: get actual list
              } 

    def __init__(self, inputdict={}, **inputvars):
        self.setup(inputdict | inputvars)
        for opt in ['MassDef', 'MassFunc', 'HaloBias', 'Concentration']: 
            try: setattr(self, opt, getattr(self, opt).lower())
            except: pass
        self.Cosmology=self.Cosmology if hasattr(self, 'Cosmology') else 'planck18'
        if self.Cosmology[0:3].lower() not in ['wma', 'eds']:
            self.Cosmology = self.Cosmology.lower()

        self.cosmology = self.colossus.cosmology.cosmology.setCosmology(self.Cosmology, self.params)
        self.params = {(k[1:] if k.startswith('_') else k):v for k, v in self.cosmology.__dict__.items()}  # set dictionary of params
        
    def H(self, zs):  # Hubble Function [km/s/Mpc], from Redshift []
        rawfunc = lambda z: self.cosmology.Hz(z=z)
        return np.vectorize(rawfunc)(zs) *u.km/u.s/u.Mpc
    
    def chi(self, zs):  # Comoving Distance [Mpc], from Redshift []
        rawfunc = lambda z: self.cosmology.comovingDistance(z_min=0, z_max=z)
        return np.vectorize(rawfunc)(zs) *u.Mpc/self.cosmology.h

    def dA(self, zs):  # Angular Distance [Mpc], from Redshift []
        dAfunc = lambda z: self.cosmology.angularDiameterDistance(z=z)
        return np.vectorize(dAfunc)(zs) *u.Mpc/self.cosmology.h

    def rhoc(self, zs):  # Critical Density [Msol/kpc^3*h^2 -> Msol/Mpc^3], from Redshift []
        rhocfunc = lambda z: self.cosmology.rho_c(z=z)
        return (np.vectorize(rhocfunc)(zs) *u.Msun/u.kpc**3*self.cosmology.h**2).to(u.Msun/u.Mpc**3)
    
    def rhom(self, zs):  # Mean Matter Density [Msol/kpc^3*h^2 -> Msol/Mpc^3], from Redshift []
        return self.rhoc(zs) * self.params['Om0']
    
    def Plin(self, ks, zs):  # Linear Matter Power Spectrum [h^3/Mpc^3 --> Msol/Mpc^3], from Wavenumber [1/Mpc --> h/Mpc] and Redshift []
        self.require(['PlinMod'])
        Plinfunc = lambda k, z: self.cosmology.matterPowerSpectrum(k=k, z=z, model=self.PlinMod)
        return np.vectorize(Plinfunc)(ks/self.cosmology.h, zs) /self.cosmology.h**3 *u.Mpc**3
    
    def bh(self, zs, logMs):
        self.require(['MassDef', 'HaloBias'])
        bhattr = self.colossus.lss.bias.haloBias
        bhfunc = lambda z, M: bhattr(M, z=z, model=self.HaloBias, mdef=self.MassDef)
        return np.vectorize(bhfunc)(zs, 10**logMs*self.cosmology.h) *u.dimensionless_unscaled
    
    def dndlogm(self, zs, logMs):
        self.require(['MassDef', 'MassFunc'])
        dndlogmattr = self.colossus.lss.mass_function.massFunction
        dndlogmfunc = lambda z, M: np.vectorize(dndlogmattr)(M, z=z, model=self.MassFunc, mdef=self.MassDef, q_out='dndlnM')
        dlogMfac = lambda Ms: (np.log(Ms[1])-np.log(Ms[0]))/(np.log10(Ms[1])-np.log10(Ms[0]))
        return dndlogmfunc(zs, 10**logMs*self.cosmology.h) *dlogMfac(10**logMs*self.cosmology.h) *self.cosmology.h**3 *1/u.Mpc**3/u.dex
    
    def c(self, zs, logMs):  # Concentration [], from Redshift [] and Halo Mass [Msol --> Msol/h]
        self.require(['MassDef', 'Concentration'])
        cattr = self.colossus.halo.concentration.concentration
        cfunc = lambda z, M: cattr(M, z=z, model=self.Concentration, mdef=self.MassDef)
        return np.vectorize(cfunc)(zs, 10**logMs*self.cosmology.h) *u.dimensionless_unscaled
    
    def logM_conv(self, zs, logMs, newmdef, conc=None):
        self.require(['MassDef', 'Concentration'])
        self.check_inputs({'MassDef': newmdef}, self.options)
        Mconvattr = self.colossus.halo.mass_defs.changeMassDefinition
        if conc is not None:
            Mconvfunc = lambda z, logM, conc: np.vectorize(Mconvattr)(10**logM*self.cosmology.h, conc(z, logM), z, self.MassDef, newmdef)
            return lambda conc: np.log10(Mconvfunc(zs, logMs, conc)[0])
        else:
            Mconvfunc = lambda z, logM: np.vectorize(Mconvattr)(10**logM*self.cosmology.h, self.c(z, logM).value, z, self.MassDef, newmdef)
            return np.log10(Mconvfunc(zs, logMs)[0])




# class hmf_model(BaseModel):  # https://hmf.readthedocs.io/en/latest/index.html
#     # TODO: in progress
#     import hmf  # Import package only if using this model
#     hmf = hmf
    
#     Parameters = list({i for k in cosmology.cosmologies.values() for i in k.keys()})  # cosmological parameters
#     options={
#         # 'Cosmology': list(cosmology.cosmologies.keys()),
#         'MassFunc': list(hmf.mass_function.FittingFunction.get_models().keys()),
#         # 'HaloBias': list(colossus.lss.bias.models.keys()),
#         # 'Concentration': list(colossus.halo.concentration.models.keys()),
#         # 'PlinMod': list(colossus.cosmology.cosmology.power_spectrum.models.keys()),
#         'MassDef': list(hmf.mass_definitions.MassDefinition.get_models().keys()),
#               } 



class mopc_model:
    def __init__(self):
        
        self.params = {'flat': True, 'H0': 70., 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8159, 'ns': 0.97}

        '''parameters used in Illustris TNG https://arxiv.org/pdf/1703.02970.pdf'''
        #params = {'flat': True, 'H0': 67.7, 'Om0': 0.31, 'Ob0': 0.0486, 'sigma8': 0.8159, 'ns': 0.97}
        
        self.cosmo_params = {
            'Omega_m':0.25,
            'hh':0.7,
            'Omega_L':0.75,
            'Omega_b':0.044,
            'rhoc_0':2.77525e2,
            'C_OVER_HUBBLE':2997.9
        }

        self.rhocrit =  1.87847e-29 * self.cosmo_params['hh']**2
        self.Msol_cgs = 1.989e33

    def hub_func(self, z):
        '''Hubble function
        '''
        Om = self.cosmo_params['Omega_m']
        Ol = self.cosmo_params['Omega_L']
        O_tot = Om + Ol
        ans = np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)
        return ans

    def rho_cz(self, z):
        '''critical density in cgs
        '''
        Ez2 = self.cosmo_params['Omega_m']*(1+z)**3. + (1-self.cosmo_params['Omega_m'])
        return self.rhocrit * Ez2

    def ComInt(self, z):
        '''comoving distance integrand
        '''
        ans = 1.0/self.hub_func(z)
        return ans

    def ComDist(self,z):
        '''comoving distance
        '''
        Om = self.cosmo_params['Omega_m']
        Ol = self.cosmo_params['Omega_L']
        O_tot = Om + Ol
        Dh = self.cosmo_params['C_OVER_HUBBLE']/self.cosmo_params['hh']
        ans = Dh*quad(self.ComInt,0,z)[0]
        if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(self.ComInt,0,z)[0])
        if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(self.ComInt,0,z)[0])
        return ans
    
    def AngDist(self, z):
        '''angular distance
        '''
        ans = self.ComDist(z)/(1.0+z)
        return ans

    def hmf(self, m,z):
        from hmf import MassFunction
        '''Shet, Mo &  Tormen 2001'''
        Mmin = np.log10(np.min(m))
        Mmax = np.log10(np.max(m))
        return MassFunction(z=z,Mmin=Mmin, Mmax=Mmax, dlog10m=(Mmax-Mmin)/49.5, hmf_model="SMT").dndm
    
    def b(self, m,z):
        from colossus.cosmology import cosmology
        from colossus.lss import bias,peaks
        cosmology.setCosmology('myCosmo',self.params)
        '''Shet, Mo &  Tormen 2001'''
        nu = peaks.peakHeight(m, z)
        delta_c = peaks.collapseOverdensity(corrections=True, z=z)
        aa, bb, cc = 0.707, 0.5, 0.6
        return 1.+ 1./(np.sqrt(aa)*delta_c) * (np.sqrt(aa)*aa*nu**2 + np.sqrt(aa)*bb*(aa*nu**2)**(1-cc) - ((aa*nu**2)**cc/(aa*nu**2)**cc+bb*(1-cc)*(1-cc/2)))
    
    def Plin_orig(self, k,z):
        from hmf import transfer
        lnk_min = np.log(np.min(k))
        lnk_max = np.log(np.max(k))
        dlnk = (lnk_max-lnk_min)/(49.5)
        '''Eisenstein & Hu (1998)'''
        p = transfer.Transfer(sigma_8=0.8344, n=0.9624, z=z, lnk_min=lnk_min, lnk_max=lnk_max, dlnk=dlnk, transfer_model="EH")
        return p.power
    
    def r200_orig(self, M, z):
        '''radius of a sphere with density 200 times the critical density of the universe.
        Input mass in solar masses. Output radius in cm.
        '''
        M_cgs = M*self.Msol_cgs
        om = self.cosmo_params['Omega_m']
        ol = self.cosmo_params['Omega_L']
        Ez2 = om * (1 + z)**3 + ol
        ans = (3 * M_cgs / (4 * np.pi * 200.*self.rhocrit*Ez2))**(1.0/3.0)
        return ans

    def H(self, zs):
        return self.hub_func(zs)*self.cosmo_params['hh']*100*u.km/u.s/u.Mpc
    
    def chi(self, zs):
        return np.vectorize(self.ComDist)(zs) * u.Mpc
    
    def dA(self, zs):
        return np.vectorize(self.AngDist)(zs) * u.Mpc
    
    def rhoc(self, zs):
        return (np.vectorize(self.rho_cz)(zs) * u.g/u.cm**3).to(u.Msun/u.Mpc**3)

    def rhoc(self, zs):
        return (np.vectorize(self.rho_cz)(zs) * u.g/u.cm**3).to(u.Msun/u.Mpc**3)
    
    def bh(self, zs, logMs):
        return np.vectorize(self.b)(10**logMs, zs)
    
    def r200c(self, zs, logMs):
        return (np.vectorize(self.r200_orig)(10**logMs, zs)*u.cm).to(u.Mpc)
    
    def dndlogm(self, zs, logMs):
        return np.vectorize(self.hmf)(10**logMs, zs)
    
    def Plin(self, ks, zs):
        return self.Plin_orig(ks, zs)


# class hmf_model(BaseModel):  # https://hmf.readthedocs.io/en/latest/index.html
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