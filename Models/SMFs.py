"""
Collections of galaxy distributions in stellar mass and redshift for various survey samples.
Classes should have galaxy number density 2D arrays over stellar and redshift and the corresponding stellar mass/redshift arrays, in consistent units.
Some samples may need to combine a SMF from one study and a redshift distribution from another.
If available, using halo masses in m200c instead of stellar masses is fine, then just don't use a SHMR later.
"""

from Basics import *

class BaseSMF:
    # Checks if the model specification is in the list
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s").keys(): 
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s').keys()}")
            else:
                setattr(self, mname, spefs[mname])
    
    # Get a distribution of N values from a catalog
    def smf_from_catalog(self, zsraw, logmsraw):
        self.zmin, self.zmax = np.round(np.floor(zsraw.min()/self.dz)*self.dz, 10), np.round((np.ceil(zsraw.max()/self.dz)+1)*self.dz, 10)
        self.logmmin, self.logmmax = np.round(np.floor(logmsraw.min()/self.dlogmstar)*self.dlogmstar, 10), np.round((np.ceil(logmsraw.max()/self.dlogmstar)+1)*self.dlogmstar, 10)
        self.zbins = np.arange(self.zmin, self.zmax, self.dz)
        self.logmstarbins = np.arange(self.logmmin, self.logmmax, self.dlogmstar)
        
        smf, zs, logms = np.histogram2d(zsraw, logmsraw, bins=[self.zbins, self.logmstarbins])
        return smf, zs[:-1]+self.dz/2, logms[:-1]+self.dlogmstar/2
    
    # Volume to convert from typical SMF units of Mpc^-3 to pure number counts
    def volumes(self, hh, T_CMB, Omega_m, Omega_L, Omega_b, **kwargs):
        cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)

        vols = np.array([(cosmo.comoving_volume(z+self.dz/2).value-cosmo.comoving_volume(z-self.dz/2).value)/(1+z)**3 for z in self.z])

        return vols * (self.area/(4*np.pi*(180/np.pi)**2))
    
    def reweight_on_dist(self, z1, N1, z2, N2):  # If you want to reweight N1 by some array N2
        if z1[1]-z1[0]>=z2[1]-z2[0]:  zf, Nf, zc, Nc = z1, N1, z2, N2
        else: zf, Nf, zc, Nc = z2, N2, z1, N1
        
        binidxs, binvalid = np.digitize(zf, zc)-1, (zf>=zc[0]) & (zf<=zc[-1])
        Nfc = np.bincount(binidxs[binvalid], weights=Nf[binvalid], minlength=zc.size)
        
        Nc_adj = np.where(np.isinf(Nfc/Nc), 0, Nfc/Nc)
        Nf_adj = np.where(np.isinf((Nc/Nfc)[binidxs]), 0, (Nc/Nfc)[binidxs])
        
        return 

        
        
    
    
class DESILRGsCrossCorr(BaseSMF):
    zbins = {'all', '1', '2', '3', '4'}
    hemispheres = {'combined':'', 'north':'', 'south':''}
    
    zdistfile="/global/cfs/projectdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/redshift_dist/main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"
    area = 140  # Check this
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['zbin', 'hemipshere'])
        
        self.zdfdata = pd.read_csv(self.zdistfile, sep=" ", skiprows=1, names=pd.read_csv(self.zdistfile, sep=" ").columns[1:])
        self.z, self.dz = (self.zdfdata.zmax+self.zdfdata.zmin)/2, self.zdfdata.zmax[1]-self.zdfdata.zmin[1]
        self.zbins = np.concatenate([self.zdfdata.zmin, self.zdfdata.zmax[-1:]])
        
        self.N_deg2 = self.zdfdata[f"{self.zbin}_{self.hemisphere}"]
        
    def N(self):
        return self.N_deg2*self.area
        


    
class DESI1Percent(BaseSMF):  # DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)
    samples = {'LRG':'', 'ELG':''}
    
    path = "/pscratch/sd/c/cpopik/save_data_point_DESI-2023-0213"
    area = 140  # TODO: Check this
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])

        self.dz = 0.1
        if self.sample=='LRG': self.zbins = np.arange(0.4, 1.2, self.dz)
        elif self.sample=='ELG': self.zbins = np.arange(0.6, 1.6, self.dz)
        self.z = self.zbins[:-1] + self.dz/2

        self.logmstar = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.dlogmstar = self.logmstar[1]-self.logmstar[0]
        self.logmstarbins = np.arange(self.logmstar[0]-self.dlogmstar/2, self.logmstar[-1]+self.dlogmstar, self.dlogmstar)
        
        self.dndlogmstar_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in self.zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]

    def dndlogmstar(self, **cosmopars):
        return cosmopars['hh']**3*self.dndlogmstar_h3
    
    def N(self, **cosmopars):
        return self.dndlogmstar_h3*cosmopars['hh']**3*self.dlogmstar*self.volumes(**cosmopars)[:, None]
        
        
class BOSSDR10(BaseSMF):  # arxiv.org/abs/1307.7735
    galaxys = {'CMASS':7, 'LOWZ':0}
    # All possible options for the group models
    groups = {'portsmouth':'LOGMASS', 'wisconsin':'MSTELLAR_MEDIAN', 'granada':'MSTELLAR_MEDIAN'}
    IMFs = {'Kroupa':'krou', 'Salpeter':'salp'}
    templates = {'starforming':'', 'passive':''}
    pops = {'Bruzual-Charlot':'bc03', 'Maraston':'m11'}
    times = {'EarlySF':'earlyform', 'ExtendedSF':'wideform'}
    dusts = {'dust':'', 'nodust':''}
    
    path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"
    area = 6373.2  # TODO: Check this
    dz, dlogmstar = 0.01, 0.1  # These can be changed as desired

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['group', 'galaxy'])
        # Different group models require different specifications
        if self.group=='portsmouth': 
            self.checkspefs(spefs, required=['template', 'IMF'])
            self.fname = f"{self.group}_stellarmass_{self.template}_{self.IMFs[self.IMF]}-v5_5_12.fits.gz"
        elif self.group=='wisconsin': 
            self.checkspefs(spefs, required=['pop'])
            self.fname = f"{self.group}_pca_{self.pops[self.pop]}-v5_5_12.fits.gz"
        elif self.group=='granada': 
            self.checkspefs(spefs, required=['IMF', 'time', 'dust'])
            self.fname  = f"{self.group}_fsps_{self.IMFS[self.IMF]}_{self.time}_{self.dust}-v5_5_12.fits.gz"

        # Properly call the mass column and get the correct galaxy selection
        mcolname, bitmask = self.groups[self.group], self.galaxys[self.galaxy]
        self.dfdata = Table.read(f"{self.path}/{self.fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (bitmask in bits))]
        
        # Create the distribution
        self.Ndist, self.z, self.logmstar = self.smf_from_catalog(zsraw=self.dfdata.Z, logmsraw=self.dfdata.LOGM)

    def dndlogmstar(self, **cosmopars):
        return self.Ndist/self.volumes(**cosmopars)[:, None]/self.dlogmstar
    
    def N(self, **cosmopars):
        return self.Ndist
        



    
    
# class Zhou2022:  # DESI LRGs (arxiv.org/abs/2208.08515)
#     path = "/global/cfs/projectdirs/desi/public/ets/vac/stellar_mass/v1/"


# class Amodeo2021():
#     mfile = "/global/homes/c/cpopik/Capybara/Data/mass_distrib.txt"
#     def __init__(self):
#         self.logmraws = np.log10(np.genfromtxt(self.mfile))
#         mbins = np.arange((np.floor(self.logmraws.min()*10)/10).round(1), (np.ceil(self.logmraws.max()*10)/10+0.1).round(1), 0.1)
#         self.SMF, self.mstars = np.histogram(self.logmraws, bins=mbins)
#         self.SMF = self.SMF[None, :]
#         self.zs = np.array([0.55])


# class Amodeo2021():
#     # These are how the centers are determind in mop-c-gt and SOLikeT_szlike, then uses Kravstov2014 Mvir_scatter
#     logmstarbins = np.linspace(9.42, 11.73, 10)
#     logmstar = (logmstarbins[1:]+logmstarbins[:-1])/2
    
#     def __init__(self):
#         pass