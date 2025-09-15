"""
Collection of Halo Mass Function obtained either from halo model codes or loaded from a data files. 
Classes should contain halo number density 2D arrays over halo mass (in m200c) and redshift and the corresponding mass/redshift arrays, in consistent units. If using functions, require input halo mass/redshift arrays to ensure output consistency between classes.
"""


import numpy as np
import astropy


class BASEHMF:
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):  # Check if the model is in the list of models
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])


class pyccl(BASEHMF):
    mfuncs = ['Angulo12', 'Bocquet16', 'Bocquet20', 'Despali16', 'Jenkins01', 'Nishimichi19', 'Press74', 'Sheth99', 'Tinker08', 'Tinker10', 'Watson13']
    hbiass = ['Bhattacharya11', 'Sheth01', 'Sheth99', 'Tinker10']
    mdefs = ['200c', '200m', 'vir', '500c', '500m']  # can actaully use any overdensity number for c and m
    
    def __init__(self, spefs):
        import pyccl as ccl  # Import package only if using this model
        self.ccl = ccl

        self.checkspefs(spefs, required=['mdef', 'mfunc', 'hbias'])
        
        self.hmffunc = getattr(ccl.halos.hmfunc, f"MassFunc{self.mfunc}")(mass_def=self.mdef)
        self.hbias = getattr(self.ccl.halos.hbias, f"HaloBias{self.hbias}")(mass_def=self.mdef)

    def initcosmo(self, hh, Omega_b, Omega_m, **kwargs):
        return self.ccl.Cosmology(h=hh, Omega_c=Omega_m-Omega_b, Omega_b=Omega_b, n_s=0.95, sigma8=0.8,transfer_function='bbks')

    def HMF(self, zs, logmshalo, **kwargs):
        cosmo = self.initcosmo(**kwargs)
        return np.array([self.hmffunc(cosmo=cosmo, M=10**logmshalo, a=1/(1+z))for z in zs])
    
    def bh(self, zs, logmshalo, **kwargs):
        cosmo = self.initcosmo(**kwargs)
        return np.array([self.hbias(cosmo=cosmo, M=10**logmshalo, a=1/(1+z))for z in zs])
    
    def Plin(self, ks, zs, **kwargs):
        cosmo = self.initcosmo(**kwargs)
        return np.array([cosmo.linear_matter_power(ks, a=1/(1+z))for z in zs]).T
    
    def Mconv(self, logmshalo, zs, mdefin, mdefout, **kwargs):
        cosmo = self.initcosmo(**kwargs)
        massconv = self.ccl.halos.massdef.mass_translator(mass_in=mdefin, mass_out=mdefout, concentration='Bhattacharya13')
        return np.array([np.log10(massconv(cosmo, 10**logmshalo, 1/(1+z))) for z in zs])



class hmf_package(BASEHMF):  # https://hmf.readthaedocs.io/en/latest/index.html
    mfuncs = ['Angulo', 'AnguloBound', 'Behroozi', 'Bhattacharya', 'Bocquet200cDMOnly', 'Bocquet200cHydro', 'Bocquet200mDMOnly', 'Bocquet200mHydro', 'Bocquet500cDMOnly', 'Bocquet500cHydro', 'Courtin', 'Crocce', 'FittingFunction', 'Ishiyama', 'Jenkins', 'Manera', 'PS', 'Peacock', 'Pillepich', 'Reed03', 'Reed07', 'SMT', 'ST', 'SimDetails', 'Tinker08', 'Tinker10', 'Union', 'Warren', 'Watson', 'Watson_FoF']
    mdefs = ['FOF','SOCritical','SOGeneric','SOMean','SOVirial','SphericalOverdensity']
    def __init__(self, spefs):
        import hmf  # Import package only if using this model
        self.hmf = hmf
        
        self.checkspefs(spefs, required=['mdef', 'mfunc'])

    def HMF(self, zs, logmshalo, hh, Omega_b, Omega_m, Omega_L, T_CMB, **kwargs):
        cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)
        logmshalo = logmshalo+np.log10(hh)
        dlog10m = logmshalo[1]-logmshalo[0]

        # Function only takes one z at a time so use list comprehension and then combine
        haloMFsraw = [self.hmf.MassFunction(z=z, Mmin=logmshalo.min(), Mmax=logmshalo.max(), dlog10m=dlog10m, hmf_model=self.mfunc, mdef_model=self.mdef, cosmo_model=cosmo) for z in zs]
        HMF_m_z = np.array([haloMF.dndlog10m*hh**3 for haloMF in haloMFsraw])
        return HMF_m_z




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