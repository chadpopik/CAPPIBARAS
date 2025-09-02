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
            if spefs[mname] not in getattr(self, f"{mname}s"): 
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
    
    # Get a distribution of N values from a catalog
    def smf_from_catalog(self, zsraw, logmsraw, zbins=None, logmstarbins=None):
        if zbins is None:
            dz = 0.01
            zmin = np.round(np.floor(zsraw.min()/dz)*dz, 10)
            zmax = np.round((np.ceil(zsraw.max()/dz)+1)*dz, 10)
            zbins = np.arange(zmin, zmax, dz)
            
        if logmstarbins is None:
            dlogmstar = 0.1
            logmmin = np.round(np.floor(logmsraw.min()/dlogmstar)*dlogmstar, 10)
            logmmax = np.round((np.ceil(logmsraw.max()/dlogmstar)+1)*dlogmstar, 10)
            logmstarbins = np.arange(logmmin, logmmax, dlogmstar)
        
        self.Ndist, _, _ = np.histogram2d(zsraw, logmsraw, bins=[zbins, logmstarbins])
        self.z, self.logmstar = (zbins[1:]+zbins[:-1])/2, (logmstarbins[1:]+logmstarbins[:-1])/2
    
    # Volume to convert from typical SMF units of Mpc^-3 to pure number counts
    def volumes(self, hh, T_CMB, Omega_m, Omega_L, Omega_b, **kwargs):
        cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)

        dz = (self.z[1]-self.z[0])
        vols = np.array([(cosmo.comoving_volume(z+dz/2).value-cosmo.comoving_volume(z-dz/2).value)/(1+z)**3 for z in self.z])
        return vols * (self.info['area']/(4*np.pi*(180/np.pi)**2))
    
    # Define SMF from Ndist
    def dndlogmstar(self, zbins=None, logmstarbins=None, **cosmopars):
        Ndist = self.N(zbins, logmstarbins)
        dlogmstar = self.logmstar[1]-self.logmstar[0]
        return Ndist/self.volumes(**cosmopars)[:, None]/dlogmstar
    
    # Define Ndist from SMF
    def N(self, **cosmopars):
        dlogmstar = self.logmstar[1]-self.logmstar[0]
        return self.dndlogmstar(**cosmopars)*dlogmstar*self.volumes(**cosmopars)[:, None]
    
    def N_z(self, **cosmopars):  # Redshift distribution
        return np.sum(self.N(**cosmopars), axis=1)
    
    def N_m(self, **cosmopars):  # Mass distribution
        return np.sum(self.N(**cosmopars), axis=0)
    
    def dndlogmstar_avez(self, **cosmopars):  # Redshift averaged SMF
        return np.average(self.dndlogmstar(**cosmopars), weights=self.N_z(**cosmopars), axis=0)

        

class DESILRGsCrossCorr(BaseSMF):  # DESI LS DR9 LRG sample from cross-correlations (arxiv.org/abs/2309.06443)
    info = {'area': 16700,  # Imaging coverage after applying masks and footprint trimming
            }
    pzbins = ['all', '1', '2', '3', '4']
    hemispheres = ['combined', 'north', 'south']
    samples = ['main', 'extended']
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['pzbin', 'hemisphere', 'sample'])
        
        zdistfile = f"{datapath}/Zhou2023B/{self.sample}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"
        cols = pd.read_csv(zdistfile, sep=" ", nrows=1).columns[1:]
        self.zdfdata = pd.read_csv(zdistfile, sep=" ", skiprows=1, names=cols)
        self.z = (self.zdfdata.zmax+self.zdfdata.zmin).values/2
        
        pzstr = f'bin_{self.pzbin}' if self.pzbin!='all' else 'all'
        self.Nz_deg2= self.zdfdata[f"{pzstr}_{self.hemisphere}"].values
        
    def N_z(self):        
        return self.Nz_deg2 * self.info['area']


class DESI1Percent(BaseSMF):  # DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)
    info = {'area': 140,# covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
            }
    
    samples = ['LRG', 'ELG']
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])

        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z = (zbins[1:]+zbins[:-1])/2

        path = f"{datapath}/Gao2023"
        self.logmstar = pd.read_csv(f"{path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        
        self.dndlogmstar_h3 = np.array([pd.read_csv(f"{path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]

    def dndlogmstar(self, **cosmopars):
        return cosmopars['hh']**3*self.dndlogmstar_h3


class BOSSDR10(BaseSMF):  # arxiv.org/abs/1307.7735
    info = {'area': 6373.2,  # TODO: Check this
            }
    
    galaxys = ['CMASS', 'LOWZ']
    # All possible options for the group models
    groups = ['portsmouth', 'wisconsin', 'granada']
    IMFs = ['Kroupa', 'Salpeter']
    templates = ['starforming', 'passive']
    pops = ['Bruzual-Charlot', 'Maraston']
    times = ['EarlySF', 'ExtendedSF']
    dusts = ['dust', 'nodust']

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['group', 'galaxy'])
        
        path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"
        # Different group models require different specifications
        if self.group=='portsmouth': 
            self.checkspefs(spefs, required=['template', 'IMF'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            fname = f"{self.group}_stellarmass_{self.template}_{imfstr}-v5_5_12.fits.gz"
        elif self.group=='wisconsin': 
            self.checkspefs(spefs, required=['pop'])
            popstr ={'Bruzual-Charlot':'bc03', 'Maraston':'m11'}[self.pop]
            fname = f"{self.group}_pca_{popstr}-v5_5_12.fits.gz"
        elif self.group=='granada': 
            self.checkspefs(spefs, required=['IMF', 'time', 'dust'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            timestr = {'EarlySF':'earlyform', 'ExtendedSF':'wideform'}[self.time]
            fname  = f"{self.group}_fsps_{imfstr}_{timestr}_{self.dust}-v5_5_12.fits.gz"
        
        mcolname = {'portsmouth':'LOGMASS', 'wisconsin':'MSTELLAR_MEDIAN', 'granada':'MSTELLAR_MEDIAN'}[self.group]
        self.dfdata = Table.read(f"{path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})
        
        # Properly call the mass column and get the correct galaxy selection
        bitmask = {'CMASS':7, 'LOWZ':0}[self.galaxy]
        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (bitmask in bits))]

    def N(self, zbins=None, logmstarbins=None, **cosmopars):  # Create the distribution
        self.smf_from_catalog(self.dfdata.Z, self.dfdata.LOGM, zbins, logmstarbins)
        return self.Ndist