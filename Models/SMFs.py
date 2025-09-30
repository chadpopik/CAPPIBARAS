"""
Collections of galaxy distributions in stellar mass and redshift for various survey samples.
Classes should have galaxy number density 2D arrays over stellar and redshift and the corresponding stellar mass/redshift arrays, in consistent units.
Some samples may need to combine a SMF from one study and a redshift distribution from another.
If available, using halo masses in m200c instead of stellar masses is fine, then just don't use a SHMR later.

TODO 1: Check SDSS DR10 area and bitmask
"""

import os
import numpy as np
import pandas as pd
import astropy
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator

datapath = "/global/homes/c/cpopik/CAPPIBARAS/Data"

class BaseSMF:
    # Checks if the model specification is in the list
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"): 
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
                
    # Calculate volume of redshift bins to convert from typical SMF density units to pure number counts
    def volumes(self, hh, T_CMB, Omega_m, Omega_L, Omega_b, zs=None, **kwargs):
        if zs is None: zs = self.z
        cosmo = astropy.cosmology.LambdaCDM(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ode0=Omega_L, Ob0=Omega_b)  # Setup astropy cosmology
        dz = (zs[1]-zs[0])  # Redshift slices
        vol = lambda z: (cosmo.comoving_volume(z+dz/2).value-cosmo.comoving_volume(z-dz/2).value)  # Comoving volume of a shell
        vols = np.array([vol(z)/(1+z)**3 for z in zs])  # Calculate for every z and convert for all z
        return vols * (self.info['area']/(4*np.pi*(180/np.pi)**2))  # Multiply by sky fraction of survey

    # Using a Stellar Halo Mass Relation, convert a Stellar Mass Function into a Halo Mass Function
    def hmf_from_smf(self, SHMR, logmstar=None, zs=None, dndlogmstar=None, **cosmopars):
        if logmstar is None: logmstar = self.logmstar
        if zs is None: zs = self.z
        if dndlogmstar is None: dndlogmstar = self.dndlogmstar(**cosmopars)
        logmhalo = np.linspace(SHMR(logmstar.min()), SHMR(logmstar.max()), logmstar.size)  # equally space halo bins
        mh_from_ms = SHMR(logmstar)  # Calculate corresponding halo masses from stellar masses
        dndlogmstar_interp = np.array([np.interp(logmhalo, mh_from_ms, dndlogmstar[i]) for i in range(zs.size)])  # interpolate to desired halo masses
        conv_fac = np.nan_to_num((np.trapz(dndlogmstar, logmstar)/np.trapz(dndlogmstar_interp, logmhalo)), nan=0.0)
        dndlogmhalo = dndlogmstar_interp*conv_fac[:, None]
        return logmhalo, dndlogmhalo

    # Create a 2D array of galaxy count binned by redshift and mass from a catalog
    def bin_catalog(self, zsraw, logmsraw, zbins, logmbins):
        if zbins is None:  # default z bins
            dz = 0.01
            zmin = np.round(np.floor(self.dfdata.Z.min()/dz)*dz, 10)
            zmax = np.round((np.ceil(self.dfdata.Z.max()/dz)+1)*dz, 10)
            zbins = np.arange(zmin, zmax, dz)

        if logmbins is None:  # default mass bins
            dlogm = 0.1
            logmmin = np.round(np.floor(self.dfdata.LOGM.min()/dlogm)*dlogm, 10)
            logmmax = np.round((np.ceil(self.dfdata.LOGM.max()/dlogm)+1)*dlogm, 10)
            logmbins = np.arange(logmmin, logmmax, dlogm)

        Ndist, _, _ = np.histogram2d(zsraw, logmsraw, bins=[zbins, logmbins])  # Bin catalog into 2D z/m array
    
        # Define zs and helpful distribution functions
        self.z = (zbins[1:]+zbins[:-1])/2  # Center of redshift bins
        logm = (logmbins[1:]+logmbins[:-1])/2  # Center of mass bins
        dndlogm = lambda **cosmopars: Ndist /self.volumes(**cosmopars)[:, None]/(logm[1]-logm[0])  # mass function [Mpc^-3 M^-1]
        return logm, dndlogm



class DESI_LRGs_XCorr(BaseSMF):  # DESI LS DR9 LRG sample from cross-correlations (Zhou+ 2023, arxiv.org/abs/2309.06443)
    info = {'area': 16700,  # Imaging coverage after applying masks and footprint trimming
            }
    pzbins = ['all', '1', '2', '3', '4']  # photo-z bin
    hemispheres = ['combined', 'north', 'south']  # sky hemisphere
    samples = ['main', 'extended']  # LRG sample
    path = f"{datapath}/Zhou2023B/"
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['pzbin', 'hemisphere', 'sample'])
        self.info['surf_dens'] = {'main':600, 'extended':1669}[self.sample]  # [deg^-2]
        self.info['comov_n_dens'] = {'main':5e-4, 'extended':1.5e-3}[self.sample]  # [h^3Mpc^-3], extended is a max value

        zdistfile = f"{self.path}/{self.sample}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"  # open data file
        cols = pd.read_csv(zdistfile, sep=" ", nrows=1).columns[1:]  # get columns from first row
        self.zdfdata = pd.read_csv(zdistfile, sep=" ", skiprows=1, names=cols)  # format into dataframe

        self.z = (self.zdfdata.zmax+self.zdfdata.zmin).values/2  # calculate z bin centers
        pzstr = f'bin_{self.pzbin}' if self.pzbin!='all' else 'all'  # get name of column base on bin
        self.Nz_deg2 = self.zdfdata[f"{pzstr}_{self.hemisphere}"].values  # get raw values from plot
        
        self.desp1P = DESI_1P({'sample':'LRG'})  # For the SMF, use DESI 1% LRG values
        self.logmstar = self.desp1P.logmstar
    
   # Stealing SMF from DESI 1%, but normalizing to the redshift distribution of XCorr LRGs
    def dndlogmstar(self, **cosmopars):
        n_z_XCorr = self.Nz_deg2 * self.info['area']/self.volumes(**cosmopars) # get zdist from XCorrLRGs
        dndlogmstar_1p = self.desp1P.dndlogmstar(**cosmopars)  # get SMF from DESI 1%
        smf = np.array([np.interp(self.z, self.desp1P.z, dndlogmstar_1p[:,i]) for i in range(dndlogmstar_1p.shape[1])]).T  # DESI 1% SMF interpolated to XCorrLRGs z values
        n_z_1p = np.trapz(self.desp1P.dndlogmstar(**cosmopars), self.desp1P.logmstar)  # zdist of DESI 1%
        zfac = n_z_XCorr/np.interp(self.z, self.desp1P.z, n_z_1p)  # normalization factor to match zdist
        return zfac[:, None]*smf



class DESI_1P(BaseSMF):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    info = {'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
            }
    
    samples = ['LRG', 'ELG']  # Galaxy Sample
    path = f"{datapath}/Gao2023"
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z = (zbins[1:]+zbins[:-1])/2

        # Read the plot data from the files
        self.logmstar = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.dndlogmstar_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]

    # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    def dndlogmstar(self, **cosmopars):
        return cosmopars['hh']**3*self.dndlogmstar_h3


class BOSS_DR10(BaseSMF):  # (Ahn+ 2013, arxiv.org/abs/1307.7735)
    info = {'area': 6373.2,  # TODO 1: Check this
            }
    
    galaxys = ['CMASS', 'LOWZ']  # Galaxy sample
    # Group models and all possible specificationss
    groups = ['portsmouth', 'wisconsin', 'granada']
    IMFs = ['Kroupa', 'Salpeter']
    templates = ['starforming', 'passive']
    pops = ['Bruzual-Charlot', 'Maraston']
    times = ['EarlySF', 'ExtendedSF']
    dusts = ['dust', 'nodust']
    
    # Path of the data in NERSC
    path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"

    def __init__(self, spefs):
        # Path through a URL if not in NERSC
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

    # Create the distribution from the dataframe
    def make_SMF(self, zbins=None, logmstarbins=None, **kwargs):
       self.logmstar, self.dndlogmstar = self.bin_catalog(self.dfdata.Z, self.dfdata.LOGM, zbins, logmstarbins)
        

class Jenna_Catalog(BaseSMF):
    info = {'area': 16700,  # assuming the same as XCorr LRGs
            }
    path = "/global/homes/c/cpopik/Data/"  # location of data
    masstypes = ['Mstar', 'M200c', 'Mvir']  # Mass type (column names)
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['masstype'])  # check for a valid mass type and set as class attribute
        self.dfdata = pd.read_csv(f"{self.path}/ACT_DR6_DESI_Y1Iron_LRGs_valid.csv")  # import datafarme

    def make_SMF(self, zbins=None, logmstarbins=None, **kwargs): 
        logm, dndlogm = self.bin_catalog(self.dfdata.z, np.log10(self.dfdata[self.masstype]), zbins, logmstarbins)  # bins the catalogs into 2D array and calculates the mass function
        if self.masstype=='Mstar': self.logmstar, self.dndlogmstar = logm, dndlogm  # if stellar mass, define the arrays as star
        else: self.logmhalo, self.dndlogmhalo = logm, dndlogm  # if stellar mass, define the arrays as halo