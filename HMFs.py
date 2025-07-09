"""
Collection of Halo Mass Function obtained either from halo model codes or loaded from a data files. 
Classes should contain halo number density 2D arrays over halo mass (in m200c) and redshift and the corresponding mass/redshift arrays, in consistent units. If using functions, require input halo mass/redshift arrays to ensure output consistency between classes.
"""

from Basics import *


class BASEHMF:
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):  # Check if the model is in the list of models
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
        
        
class pyccl(BASEHMF):
    mfuncs = ['Angulo12', 'Bocquet16', 'Bocquet20', 'Despali16', 'Jenkins01', 'Nishimichi19', 'Press74', 'Sheth99', 'Tinker08', 'Tinker10', 'Watson13']
    mdefs = ['200c', '200m', 'vir', '500c', '500m']  # can actaully use any overdensity number for c and m
    def __init__(self, spefs):
        import pyccl as ccl  # Import package only if using this model
        self.ccl = ccl

        self.checkspefs(spefs, required=['mdef', 'mfunc'])
        
        self.hmffunc = getattr(ccl.halos.hmfunc, f"MassFunc{self.mfunc}")(mass_def=self.mdef)
        
    def HMF(self, ms, zs, cosmopars):
        self.cosmo = self.ccl.Cosmology(h=cosmopars['hh'], Omega_c=cosmopars['Omega_m']-cosmopars['Omega_b'], Omega_b=cosmopars['Omega_b'], n_s=0.95, sigma8=0.8,transfer_function='bbks')
                
        HMF_m_z = np.array([self.hmffunc(cosmo=self.cosmo, M=ms, a=1/(1+z))for z in zs])
        return HMF_m_z


class hmf_package(BASEHMF):  # https://hmf.readthaedocs.io/en/latest/index.html
    mfuncs = ['Angulo', 'AnguloBound', 'Behroozi', 'Bhattacharya', 'Bocquet200cDMOnly', 'Bocquet200cHydro', 'Bocquet200mDMOnly', 'Bocquet200mHydro', 'Bocquet500cDMOnly', 'Bocquet500cHydro', 'Courtin', 'Crocce', 'FittingFunction', 'Ishiyama', 'Jenkins', 'Manera', 'PS', 'Peacock', 'Pillepich', 'Reed03', 'Reed07', 'SMT', 'ST', 'SimDetails', 'Tinker08', 'Tinker10', 'Union', 'Warren', 'Watson', 'Watson_FoF']
    mdefs = ['FOF','SOCritical','SOGeneric','SOMean','SOVirial','SphericalOverdensity']
    def __init__(self, spefs):
        import hmf  # Import package only if using this model
        self.hmf = hmf
        
        self.checkspefs(spefs, required=['mdef', 'mfunc'])

    def HMF(self, ms, zs, cosmopars):
        cosmo = astropy.cosmology.LambdaCDM(H0=cosmopars["hh"]*100, Tcmb0=2.726, Om0=cosmopars["Omega_m"], Ode0=cosmopars["Omega_L"], Ob0=cosmopars["Omega_b"])
        
        # Set spacing between logMhalo values, which must be logspaced
        try: dlog10m = np.min(np.log10(ms[1:])-np.log10(ms[:-1]))  # TODO 3
        except ValueError: dlog10m = 0.1

        # Function only takes one z at a time so use list comprehension and then combine
        haloMFsraw = [self.hmf.MassFunction(z=z, Mmin=np.log10(np.min(ms))-dlog10m, Mmax=np.log10(np.max(ms))+dlog10m, dlog10m=dlog10m, hmf_model=self.mfunc, mdef_model=self.mdef, cosmo_model=cosmo) for z in zs]  # TODO 2
        HMF_m_z = np.array([np.interp(ms, haloMF.m, haloMF.dndlog10m) for haloMF in haloMFsraw]).T
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