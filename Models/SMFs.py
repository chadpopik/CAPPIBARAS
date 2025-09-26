"""
Collections of galaxy distributions in stellar mass and redshift for various survey samples.
Classes should have galaxy number density 2D arrays over stellar and redshift and the corresponding stellar mass/redshift arrays, in consistent units.
Some samples may need to combine a SMF from one study and a redshift distribution from another.
If available, using halo masses in m200c instead of stellar masses is fine, then just don't use a SHMR later.

TODO 1: Add DESI SMF from catalog
"""

datapath = "/global/homes/c/cpopik/CAPPIBARAS/Data/"

import os
import numpy as np
import pandas as pd
import astropy
from astropy.table import Table

class BaseSMF:
    # Checks if the model specification is in the list
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"): 
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
                
    # Calculate volume of redshift bins to convert from typical SMF density units to pure number counts
    def volumes(self, hh, T_CMB, Omega_m, Omega_L, Omega_b, **kwargs):  # Needs cosmological parameters
        cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)  # Setup astropy cosmology
        dz = (self.z[1]-self.z[0])  # Redshift slices
        vol = lambda z: (cosmo.comoving_volume(z+dz/2).value-cosmo.comoving_volume(z-dz/2).value)  # Comoving volume of a shell
        vols = np.array([vol(z)/(1+z)**3 for z in self.z])  # Calculate for every z and convert for all z
        return vols * (self.info['area']/(4*np.pi*(180/np.pi)**2))  # Multiply by sky fraction of survey
    
    # Create a 2D array of galaxy count binned by redshift and stellar mass from a catalog
    def bin_catalog(self, zsraw, logmsraw, zbins, logmstarbins):
        if zbins is None: zbins = self.zbins  # Default zbins
        if logmstarbins is None: logmstarbins = self.logmstarbins  # Default mstar bins
        Ndist, _, _ = np.histogram2d(zsraw, logmsraw, bins=[zbins, logmstarbins])  # Bin catalog into 2D z/m array
        self.z, self.logmstar = (zbins[1:]+zbins[:-1])/2, (logmstarbins[1:]+logmstarbins[:-1])/2  # Calculate centers of bins
        self.dndlogmstar = lambda **cosmopars: Ndist /self.volumes(**cosmopars)[:, None]/(self.logmstar[1]-self.logmstar[0])
        self.N_z_logmstar = lambda **cosmopars: Ndist
        self.N = lambda **cosmopars: np.sum(Ndist)
        self.N_z = lambda **cosmopars: np.sum(Ndist, axis=1)
        self.dNdlogmstar = lambda **cosmopars: Ndist/(self.logmstar[1]-self.logmstar[0])
        self.dNdz = lambda **cosmopars: np.sum(Ndist, axis=1)(self.z[1]-self.z[0])
    
    # Using a Stellar Halo Mass Relation, convert a Stellar Mass Function into a Halo Mass Function
    def dndlogmhalo(self, SHMR, **cosmopars):
        self.logmhalo = np.linspace(SHMR(self.logmstar.min()), SHMR(self.logmstar.max()), self.logmstar.size)  # equally space halo bins
        smf = self.dndlogmstar(**cosmopars)
        mh_from_ms = SHMR(self.logmstar)
        dlogmstar, dlogmhalo = self.logmstar[1]-self.logmstar[0], self.logmhalo[1]-self.logmhalo[0]
        dndlogmhalo = np.array([np.interp(self.logmhalo, mh_from_ms, smf[i]*dlogmstar)/dlogmhalo for i in range(self.z.size)])
        return dndlogmhalo
    
    
    # Convinient functions to have on hand when playing with distributions
    def dndzdlogmstar(self, **cosmopars):        
        return self.dndlogmstar(**cosmopars)/(self.z[1]-self.z[0])
    
    def dndz(self, **cosmopars):        
        return self.dndlogmstar(**cosmopars)*(self.logmstar[1]-self.logmstar[0])/(self.z[1]-self.z[0])
    
    def dndz_z(self, **cosmopars):        
        return np.trapz(self.dndlogmstar(**cosmopars), self.logmstar)/(self.z[1]-self.z[0])
    
    def dndlogmstar_m(self, **cosmopars):        
        return np.trapz(self.dndz(**cosmopars), self.z, axis=0)/(self.logmstar[1]-self.logmstar[0])
    
    def n(self, **cosmopars):
        return self.dndlogmstar(**cosmopars)*(self.logmstar[1]-self.logmstar[0])
    
    def n_m(self, **cosmopars):
        return self.dndlogmstar_m(**cosmopars)*(self.logmstar[1]-self.logmstar[0])
    
    def n_z(self, **cosmopars):
        return self.dndz_z(**cosmopars)*(self.z[1]-self.z[0])
    
    def ntot(self, **cosmopars):
        return np.sum(np.trapz(self.dndlogmstar(**cosmopars), self.logmstar))
    
    def dNdlogmstar(self, **cosmopars):
        return self.dndlogmstar(**cosmopars)*self.volumes(**cosmopars)[:, None]
    
    def dNdz(self, **cosmopars):        
        return self.dNdlogmstar(**cosmopars)*(self.logmstar[1]-self.logmstar[0])/(self.z[1]-self.z[0])
    
    def dNdlogmstar_m(self, **cosmopars):        
        return np.trapz(self.dNdz(**cosmopars), self.z, axis=0)/(self.logmstar[1]-self.logmstar[0])
    
    def dNdz_z(self, **cosmopars):
        return self.dndz_z(**cosmopars)*self.volumes(**cosmopars)
    
    def N(self, **cosmopars):
        return self.dNdlogmstar(**cosmopars)*(self.logmstar[1]-self.logmstar[0])
    
    def N_m(self, **cosmopars):
        return self.dNdlogmstar_m(**cosmopars)*(self.logmstar[1]-self.logmstar[0])
    
    def N_z(self, **cosmopars):
        return self.dNdz_z(**cosmopars)*(self.z[1]-self.z[0])
    
    def Ntot(self, **cosmopars):
        return np.sum(np.trapz(self.dNdlogmstar(**cosmopars), self.logmstar))
    


class DESILRGsCrossCorr(BaseSMF):  # DESI LS DR9 LRG sample from cross-correlations (arxiv.org/abs/2309.06443)
    info = {'area': 16700,  # Imaging coverage after applying masks and footprint trimming
            }
    pzbins = ['all', '1', '2', '3', '4']  # photo-z bin
    hemispheres = ['combined', 'north', 'south']
    samples = ['main', 'extended']
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['pzbin', 'hemisphere', 'sample'])

        # Open data file and load into dataframe
        zdistfile = f"{datapath}/Zhou2023B/{self.sample}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"
        cols = pd.read_csv(zdistfile, sep=" ", nrows=1).columns[1:]
        self.zdfdata = pd.read_csv(zdistfile, sep=" ", skiprows=1, names=cols)

        # Get z values and number density from plots
        self.z = (self.zdfdata.zmax+self.zdfdata.zmin).values/2
        pzstr = f'bin_{self.pzbin}' if self.pzbin!='all' else 'all'
        self.Nz_deg2 = self.zdfdata[f"{pzstr}_{self.hemisphere}"].values
        
        self.smffrom = DESI1Percent({'sample':'LRG'})
        self.logmstar = self.smffrom.logmstar
    
   # Stealing SMF from DESI 1%, but normalizing to the redshift distribution of this
    def dndlogmstar(self, **cosmopars):
        dndz_z_raw = self.Nz_deg2 * self.info['area']/self.volumes(**cosmopars) / (self.z[1]-self.z[0])
        smfraw = self.smffrom.dndlogmstar(**cosmopars)
        smf = np.array([np.interp(self.z, self.smffrom.z, smfraw[:,i]) for i in range(smfraw.shape[1])]).T
        zfac = (self.z[1]-self.z[0])*dndz_z_raw/np.interp(self.z, self.smffrom.z, np.trapz(self.smffrom.dndlogmstar(**cosmopars),self.smffrom.logmstar))
        return zfac[:, None]*smf



class DESI1Percent(BaseSMF):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    info = {'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
            }
    
    samples = ['LRG', 'ELG']  # Galaxy Sample
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        
        # Define the z bins from the plots
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z = (zbins[1:]+zbins[:-1])/2

        # Read the plot data from the files
        path = f"{datapath}/Gao2023"
        self.logmstar = pd.read_csv(f"{path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.dndlogmstar_h3 = np.array([pd.read_csv(f"{path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]

    # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    def dndlogmstar(self, **cosmopars):
        return cosmopars['hh']**3*self.dndlogmstar_h3


class BOSSDR10(BaseSMF):  # (Ahn+ 2013, arxiv.org/abs/1307.7735)
    info = {'area': 6373.2,  # TODO: Check this
            }
    
    galaxys = ['CMASS', 'LOWZ']  # Galaxy sample
    # Group models and all possible specificationss
    groups = ['portsmouth', 'wisconsin', 'granada']
    IMFs = ['Kroupa', 'Salpeter']
    templates = ['starforming', 'passive']
    pops = ['Bruzual-Charlot', 'Maraston']
    times = ['EarlySF', 'ExtendedSF']
    dusts = ['dust', 'nodust']

    def __init__(self, spefs):
        # Path of the data in NERSC, or path through a URL if not in NERSC
        self.path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"
        if os.path.isdir(self.path): pass
        else: self.path = "https://data.sdss.org/sas/dr10/boss/spectro/redux/galaxy/v1_0/"

        self.checkspefs(spefs, required=['group', 'galaxy'])
        
        # Each group model needs different specifications and has a different naming scheme
        if self.group=='portsmouth': 
            self.checkspefs(spefs, required=['template', 'IMF'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            fname = f"{self.group}_stellarmass_{self.template}_{imfstr}-v5_5_12.fits.gz"
        elif self.group=='wisconsin': 
            self.checkspefs(spefs, required=['pop'])
            popstr = {'Bruzual-Charlot':'bc03', 'Maraston':'m11'}[self.pop]
            fname = f"{self.group}_pca_{popstr}-v5_5_12.fits.gz"
        elif self.group=='granada': 
            self.checkspefs(spefs, required=['IMF', 'time', 'dust'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            timestr = {'EarlySF':'earlyform', 'ExtendedSF':'wideform'}[self.time]
            fname  = f"{self.group}_fsps_{imfstr}_{timestr}_{self.dust}-v5_5_12.fits.gz"
        
        # Fetch the data with properly naming and renaming the mass column
        mcolname = {'portsmouth':'LOGMASS', 'wisconsin':'MSTELLAR_MEDIAN', 'granada':'MSTELLAR_MEDIAN'}[self.group]
        self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})
        
        # Select the correct galaxy sample using the bitmasks
        bitmask = {'CMASS':7, 'LOWZ':0}[self.galaxy]
        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (bitmask in bits))]
        
        # Set a default binning in redshift and stellar mass
        dz = 0.01
        zmin = np.round(np.floor(self.dfdata.Z.min()/dz)*dz, 10)
        zmax = np.round((np.ceil(self.dfdata.Z.max()/dz)+1)*dz, 10)
        self.zbins = np.arange(zmin, zmax, dz)
        self.z = (self.zbins[1:]+self.zbins[:-1])/2
            
        dlogmstar = 0.1
        logmmin = np.round(np.floor(self.dfdata.LOGM.min()/dlogmstar)*dlogmstar, 10)
        logmmax = np.round((np.ceil(self.dfdata.LOGM.max()/dlogmstar)+1)*dlogmstar, 10)
        self.logmstarbins = np.arange(logmmin, logmmax, dlogmstar)
        self.logmstar = (self.logmstarbins[1:]+self.logmstarbins[:-1])/2

    # Create the distribution from the dataframe
    def make_SMF(self, zbins=None, logmstarbins=None, **cosmopars):
        self.bin_catalog(self.dfdata.Z, self.dfdata.LOGM, zbins, logmstarbins)
        
class Jennastuff(BaseSMF):
    info = {'area': 16700,  # Imaging coverage after applying masks and footprint trimming
            }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=[])

        self.dfdata = pd.read_csv("/global/homes/c/cpopik/Data/ACT_DR6_DESI_Y1Iron_LRGs_valid.csv")
        
        dz = 0.01
        zmin = np.round(np.floor(self.dfdata.z.min()/dz)*dz, 10)
        zmax = np.round((np.ceil(self.dfdata.z.max()/dz)+1)*dz, 10)
        self.zbins = np.arange(zmin, zmax, dz)
        self.z = (self.zbins[1:]+self.zbins[:-1])/2
            
        dlogmstar = 0.1
        logmmin = np.round(np.floor(np.log10(self.dfdata.Mstar.min())/dlogmstar)*dlogmstar, 10)
        logmmax = np.round((np.ceil(np.log10(self.dfdata.Mstar.max())/dlogmstar)+1)*dlogmstar, 10)
        self.logmstarbins = np.arange(logmmin, logmmax, dlogmstar)
        self.logmstar = (self.logmstarbins[1:]+self.logmstarbins[:-1])/2

    # Create the distribution from the dataframe
    def make_SMF(self, zbins=None, logmstarbins=None, **cosmopars):
        self.bin_catalog(self.dfdata.z, np.log10(self.dfdata.Mstar), zbins, logmstarbins)