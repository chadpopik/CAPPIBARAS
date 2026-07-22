"""
DEPRECIATED
"""

import os
import numpy as np
import pandas as pd
import astropy
import astropy.cosmology
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator

from Models import SHMRs  # For running on level of CAPPIABRAS, one folder up from Models where SMFs is

from config import DATA_PATH

datapath = DATA_PATH

class BaseSMF:
    # Checks if the model specification is in the list
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"): 
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
                
    # Calculate volume of redshift bins to convert from typical SMF density units to pure number counts
    def volumes(self, zs=None, hh=0.7, T_CMB=2.725, Omega_m=0.3, Omega_b=0.044, **kwargs):
        if zs is None: zs = self.z  # Use preset redshift of class if not given
        dz = (zs[1]-zs[0])  # Redshift slices
        cosmo = astropy.cosmology.Planck18.clone(H0=hh*100, Tcmb0=T_CMB, Om0=Omega_m, Ob0=Omega_b)  # Setup astropy cosmology
        vol = lambda z: (cosmo.comoving_volume(z+dz/2).value-cosmo.comoving_volume(z-dz/2).value)  # Comoving volume of a shell
        vols = np.array([vol(z)/(1+z)**3 for z in zs])  # Calculate non-comoving for every z
        return vols * (self.info['area']/(4*np.pi*(180/np.pi)**2))  # Multiply by sky fraction of survey

    # Using a Stellar Halo Mass Relation, convert a Stellar Mass Function into a Halo Mass Function
    def hmf_from_smf(self, SHMR, logmstar=None, zs=None, dndlogmstar=None, logmhalo=None, **cosmopars):
        if logmstar is None: logmstar = self.logmstar
        if zs is None: zs = self.z
        if dndlogmstar is None: dndlogmstar = self.dndlogmstar(**cosmopars)
        if logmhalo is None:
            logmhalo = np.linspace(SHMR(logmstar.min())(), SHMR(logmstar.max())(), logmstar.size)  # equally space halo bins
        mh_from_ms = SHMR(logmstar)()  # Calculate corresponding halo masses from stellar masses
        dndlogmstar_interp = np.array([np.interp(logmhalo, mh_from_ms, dndlogmstar[i]) for i in range(zs.size)])  # interpolate to desired halo masses
        conv_fac = np.nan_to_num((np.trapezoid(dndlogmstar, logmstar)/np.trapezoid(dndlogmstar_interp, logmhalo)), nan=0.0)
        dndlogmhalo = dndlogmstar_interp*conv_fac[:, None]
        return logmhalo, dndlogmhalo

    # Create a 2D array of galaxy count binned by redshift and mass from a catalog
    def bin_catalog(self, zsraw, logmsraw, zbins=None, logmbins=None):
        dz, dlogm = 0.01, 0.1  # Default spacing if bins aren't given
        if zbins is None:  # If z bins aren't given
            zmin, zmax = np.floor(zsraw.min()/dz)*dz, np.ceil(zsraw.max()/dz)*dz
            zbins = np.arange(np.round(zmin, 10), np.round(zmax, 10)+dz, dz)
        if logmbins is None:  # If mass bins aren't given
            logmmin, logmmax = np.floor(logmsraw.min()/dlogm)*dlogm, np.ceil(logmsraw.max()/dlogm)*dlogm
            logmbins = np.arange(np.round(logmmin, 10), np.round(logmmax, 10)+dlogm, dlogm)

        zs, logms = (zbins[1:]+zbins[:-1])/2, (logmbins[1:]+logmbins[:-1])/2  # Center of redshift and mass bins
        Ndist, _, _ = np.histogram2d(zsraw, logmsraw, bins=[zbins, logmbins])  # Bin catalog into 2D z/m array
        dndlogm = lambda **cosmopars: Ndist /self.volumes(**cosmopars)[:, None]/(logms[1]-logms[0])  # mass function [Mpc^-3 M^-1]
        dNdz = np.histogram(zsraw, bins=zbins)[0]/(zs[1]-zs[0])  # redshift distribution [Mpc^-3 M^-1]
        return dndlogm, lambda **kwargs: dNdz, zs, logms



class Jenna_Catalog(BaseSMF):
    info = {'area': 16700,  # assuming the same as XCorr LRGs
            }
    path = datapath  # location of data
    masstypes = ['Mstar', 'M200c', 'Mvir']  # Mass type (column names)
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['masstype'])  # check for a valid mass type and set as class attribute
        self.dfdata = pd.read_csv(f"{self.path}/ACT_DR6_DESI_Y1Iron_LRGs_valid.csv")  # import datafarme

    def make_SMF(self, zbins=None, logmstarbins=None, **kwargs): 
        logm, dndlogm = self.bin_catalog(self.dfdata.z, np.log10(self.dfdata[self.masstype]), zbins, logmstarbins)  # bins the catalogs into 2D array and calculates the mass function
        if self.masstype=='Mstar': self.logmstar, self.dndlogmstar = logm, dndlogm  # if stellar mass, define the arrays as star
        else: self.logmhalo, self.dndlogmhalo = logm, dndlogm  # if stellar mass, define the arrays as halo


class RiedGuachalla2025(BaseSMF):  # ACT DR6 maps stacked on DESI Y1 LRGs (Ried-Guachalla+ 2025, arxiv.org/abs/2503.19870)
    bins = ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4']
    path = f"{datapath}/RiedGuachalla2025"
    info = {'area': 4300,  # deg2
            'logmhalomean':13.4,}
    # Assuming zbins have same mvalues as all, and mbins have same zvalues as all
    bininfo = {
        'zmin': {'all':0.4, 'z_1':0.4, 'z_2':0.6, 'z_3':0.8, 'z_4':0.9, 'mass_1':0.4, 'mass_2':0.4, 'mass_3':0.4, 'mass_4':0.4},
        'zmax': {'all':1.1, 'z_1':0.6, 'z_2':0.8, 'z_3':0.95, 'z_4':1.1, 'mass_1':1.1, 'mass_2':1.1, 'mass_3':0.4, 'mass_4':1.1},
        'logmstarmin': {'all':10.5, 'z_1':10.5, 'z_2':10.5, 'z_3':10.5, 'z_4':10.5, 'mass_1':10.5, 'mass_2':11.2, 'mass_3':11.4, 'mass_4':11.6},
        'logmstarmax': {'all':12.5, 'z_1':12.5, 'z_2':12.5, 'z_3':12.5, 'z_4':12.5, 'mass_1':11.2, 'mass_2':11.4, 'mass_3':11.6, 'mass_4':12.5},
        'zmean': {'all':0.74, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.76, 'mass_2':0.75, 'mass_3':0.71, 'mass_4':0.69},
        'zmed': {'all':0.75, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.79, 'mass_2':0.76, 'mass_3':0.70, 'mass_4':0.67},
        'logmstarmean': {'all':2.2, 'z_1':2.4, 'z_2':2.3, 'z_3':2.0, 'z_4':2.1, 'mass_1':1.2, 'mass_2':2.0, 'mass_3':3.0, 'mass_4':5.1},  # 10e11 Mstar/Msun
        'ngal': {'all':825283, 'z_1':195877, 'z_2':235620, 'z_3':235620, 'z_4':96346, 'mass_1':244932, 'mass_2':320914, 'mass_3':194037, 'mass_4':53997},
        }

    def __init__(self, spefs):    
        self.checkspefs(spefs, required=['bin'])
        
        for prop in self.bininfo.keys():  # Set general properties of bin
            self.info[prop] = self.bininfo[prop][self.bin]
            if prop=='logmstarmean': self.info[prop] = np.log10(self.bininfo[prop][self.bin]*1e11)


        self.shmr = SHMRs.Gao2023({'sample':'Psat_Mh'}).SHMR

        dz, dlogmhalo = 0.05, 0.2
        mhalomin, mhalomax = np.round(self.shmr(self.info['logmstarmin'])(), 1), np.round(self.shmr(self.info['logmstarmax'])(), 1)
        zbins, logmhalobins = np.arange(self.info['zmin'], self.info['zmax']+dz, dz), np.arange(mhalomin, mhalomax+dlogmhalo, dlogmhalo)
        self.z, self.logmhalo = (zbins[1:]+zbins[:-1])/2, (logmhalobins[1:]+logmhalobins[:-1])/2
        
        zdf = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # FIG. 2: Redshift distribution of the DESI LRG Y1 galaxies overlapping the ACT map
        logmstardf = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # FIG. 3: The binned stellar mass distribution of the DESI LRG Y1 galaxies overlapping the ACT DR6 map
        
        if self.bin[0]=='z':
            dNdz = np.histogram(zdf[f'{self.bin}'], bins=zbins)[0]/dz
        else:
            dNdz = np.histogram(np.concatenate([zdf[f'z_{i}'] for i in range(1, 5)]), bins=zbins)[0]/dz
            
        self.dNdz = lambda **kwargs: dNdz
            
        if self.bin[0]=='m':
            logmhalos = self.shmr(np.log10(logmstardf[f'{self.bin}']))()
        else:
            logmhalos = np.concatenate([self.shmr(np.log10(logmstardf[f'mass_{i}']))() for i in range(1, 5)])
            
        self.dNdlogmhalo = np.histogram(logmhalos, bins=logmhalobins)[0]/dlogmhalo
        
    def dndlogmhalo(self, **cpars):
        norm_fac = self.dNdz()[:,None]/np.trapezoid(self.dNdlogmhalo, self.logmhalo)*(self.z[1]-self.z[0])
        return self.dNdlogmhalo * norm_fac/self.volumes(zs=self.z, **cpars)[:, None]



class Kou2023(BaseSMF):   # CMASS DR12 (Kou+ 2023, arxiv.org/abs/2211.07502)
    info = {'zmin':0.47, 'zmax':0.59, 'area':14000, 'logmstarmax':12.5}
    mbins = ['1', '2', '3', '4']
    # technically they magnitude sort into starforming and passive mass models
    mbininfo = {
        'logmstarmin': {'1':10.8, '2':11.1, '3':11.25, '4':11.4},
        'ngal': {'1':473596, '2':396298, '3':250964, '4':124493},
    }
    
    def __init__(self, spefs):  # USING DR10 but need to switch to DR12
        self.checkspefs(spefs, required=['mbin'])
        for inf in self.mbininfo.keys():
            self.info[inf] = self.mbininfo[inf][self.mbin]
            
        K18 = SHMRs.Kravstov2018({'sample':'Mvir_scatter'})
        self.info['logmhalomin'] = np.round(K18.SHMR(self.info['logmstarmin'])(), 2)
        self.info['logmhalomax'] = np.round(K18.SHMR(self.info['logmstarmax'])(), 2)
        
        zbins = np.linspace(self.info['zmin'], self.info['zmax'], 11)
        logmhalobins = np.linspace(self.info['logmhalomin'], self.info['logmhalomax'], 21)
        BOSSDR12 = SDSSBOSS({'galaxy':'CMASS', 'group':'wisconsin', 'pop':'Maraston', 'DR':'12'})  # Might need to check these
        self.dndlogmhalo, self.dNdz, self.z, self.logmhalo = self.bin_catalog(BOSSDR12.dfdata.Z, K18.SHMR(BOSSDR12.dfdata.LOGM)(), zbins, logmhalobins)


class Liu2025(BaseSMF): # ACT DR6 maps stacked on DESI LRGs for cross-correlation (Liu+ 2025, arxiv.org/abs/2502.08850)
    path = f"{datapath}/Liu2025"
    zbins = ['1', '2', '3', '4']
    info = {'area':4100,  # extrapolated
            }
    spefinfo = {
        'ngal': {'1':332280, '2':608100, '3':671738, '4':615543},  # objects in overlap, tb1
        'ngal_unmasked': {'1':1118496, '2':2031303, '3':2240982, '4':2049158},  # total objects, tb1
    }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['zbin'])
        
        for inf in self.spefinfo.keys():
            self.info[inf] = self.spefinfo[inf][self.zbin]
        DESIXLRGs = Zhou2023({'pzbin':self.zbin, 'hemisphere':'combined','sample':'main'})
        for inf in ['zmean', 'ndens', 'logmhalomean']:  # info taken from Zhou2023
            self.info[inf] = DESIXLRGs.info[inf]
        self.info['area'] = self.info['ngal']/self.info['ndens']  # estimate from ngal and ndens values in paper

        # Loading redshift distribution from file
        zcols = pd.read_csv(f"{datapath}/Zhou2023B/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # col names from Zhou2023
        zdf = pd.DataFrame(np.genfromtxt(f"{self.path}/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"), columns=zcols)  # Spectroscopic distributions of four sub-sample photometric redshift bins
        z = (zdf.zmin+zdf.zmax).values/2

        zmin, zmax = {'1':0.25, '2':0.35, '3':0.55, '4':0.65}, {'1':0.65, '2':0.85, '3':1.05, '4':1.25}
        self.z = np.linspace(zmin[self.zbin], zmax[self.zbin], 20)  # Slim down the z distribution    
        self.dNdz = lambda **kwargs: np.interp(self.z, z, zdf[f'bin_{self.zbin}_combined'].values/(z[1]-z[0]))* self.info['area']

        made_dndlogmhalo = self.dndlogmhalo()

    def dndlogmhalo(self, **cosmopars):
        desi1p = DESI1P({'sample':'LRG'})        # Stealing SMF from DESI 1% LRG values, same with SHMR
        zs_1p, logmstar_1p, dndlogmstar_1p = desi1p.z, desi1p.logmstar, desi1p.dndlogmstar(**cosmopars)
        nz_1p = np.trapezoid(dndlogmstar_1p, logmstar_1p)  # zdist of DESI 1%
        nz_XCorr = self.dNdz() * (self.z[1]-self.z[0]) /self.volumes(**cosmopars) # get zdist from XCorrLRGs
        zfac = nz_XCorr/np.interp(self.z, zs_1p, nz_1p)  # normalization factor to match zdist
        
        logmhalos = np.linspace(11.6, 14, 20)  # slim down logmhalo distribution
        shmr = SHMRs.Gao2023({'sample':'Psat_Mh'})
        self.logmhalo, dndlogmhalo_1p = self.hmf_from_smf(shmr.SHMR, logmstar=logmstar_1p, zs=zs_1p, dndlogmstar=dndlogmstar_1p, logmhalo=logmhalos,**cosmopars)        
        dndlogmhalo_1p_interp = np.array([np.interp(self.z, zs_1p, dndlogmhalo_1p[:,i]) for i in range(dndlogmhalo_1p.shape[1])]).T  # DESI 1% HMF interpolated to XCorrLRGs z values
        return zfac[:, None]*dndlogmhalo_1p_interp


class Schaan2021(BaseSMF):  # ACT DR5 maps stacked on SDSS BOSS DR10 (Schaan+ 2021, arxiv.org/abs/2009.05557)
    info = {'mdef':'vir',  # halo mass definition, fig3
            'logmstarmax': np.round(np.log10(5.5e11), 2),  # max halo mass cut, pg10p2
            'zmin':0.4, 'zmax':0.7,  # redshift range, pg4p2
            'area':6000, # area of overlap between, TODO 1: NEEDS CHECKING
            }
    samples = ['CMASS', 'LOWZ']
    spefinfo = {
        'zmean': {'LOWZ':0.31, 'CMASS':0.54},  # mean redshift, fig2
        'logmhalomean': {'LOWZ':np.round(np.log10(5e13), 2), 'CMASS':np.round(np.log10(3e13), 2)},  # host halo virial mass, fig3
        'ngal_catalog':{'LOWZ':218905, 'CMASS': 501844, 'CMASSm':777202},  # total galaxies in BOSS catalog, pg5p2
        'ngal_overlap': {'LOWZ':151713, 'CMASS': 325518, 'CMASSm':385137},  # galaxies in ACT BOSS overlap, pg5p2
        'ngal_masked': {'LOWZ':145714, 'CMASS': 312708, 'CMASSm':368701},  # galaxies in overlap after masking, pg5p2
        'ngal': {'LOWZ':134702, 'CMASS':311309, 'CMASSm':360084},  # final galaxy count after applying upper mass limit, pg5p2
        }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        for inf in self.spefinfo.keys():
            self.info[inf] = self.spefinfo[inf][self.sample]
    
        K18 = SHMRs.Kravstov2018({'sample':'Mvir_scatter'})  # SHMR, pg10p2
        self.info['logmhalomax'] = np.round(K18.SHMR(self.info['logmstarmax'])(), 2)
        
        zbins = np.linspace(self.info['zmin'], self.info['zmax'], 11)  # Can change coarseness
        logmhalobins = np.linspace(12, self.info['logmhalomax'], 21)  # Can change coarseness, logmmin arbitrary
        BOSSDR10 = SDSSBOSS({'galaxy':self.sample, 'group':'wisconsin', 'pop':'Maraston', 'DR':'10'})  # Check spefs
        zs, logmhalos = BOSSDR10.dfdata.Z, K18.SHMR(BOSSDR10.dfdata.LOGM)()
        self.dndlogmhalo, self.dNdz, self.z, self.logmhalo = self.bin_catalog(zs, logmhalos, zbins, logmhalobins)



        






class Zhou2023(BaseSMF):  # DESI LS DR9 LRG sample from cross-correlations (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B/"

    pzbins = ['all', '1', '2', '3', '4']  # photo-z bin
    hemispheres = ['combined', 'north', 'south']  # sky hemisphere
    samples = ['main', 'extended']  # LRG sample

    info={}
    spefinfo = {
        # 'ngal': {'main':2.3e6, 'extended':22e3},  # fig2
        # 'area': {'main':{'combined':16700, 'north':4200, 'south':12500},  # tb1, pg3 p2
        #          'extended':{'combined':230, 'north':100, 'south':130}},  # tb1, pg9p1
        'area': {'main':13800, 'extended': 100},  # TODO: area we sure?
        'ndens': {'main': {'all':600, '1': 81.9, '2': 148.1, '3': 162.4, '4': 148.3},  # Table 1 & 2
                'extended': {'all':1669, '1': 185.5, '2': 311.0, '3': 422.6, '4': 438.4},},  # Table 1 & 3
        'zmean': {'main': {'1': 0.470, '2': 0.628, '3': 0.791, '4': 0.924},  # Table 2
                'extended': {'1': 0.467, '2': 0.633, '3': 0.794, '4': 0.929},},  # Table 3
        'pzmin': {'main': {'north': {'all':0.400, '1': 0.400, '2': 0.545, '3': 0.719, '4': 0.851},   # Table 2
                        'south': {'all':0.400, '1': 0.400, '2': 0.540, '3': 0.713, '4': 0.860},},   # Table 2
                'extended': {'north': {'all':0.400, '1': 0.400, '2': 0.545, '3': 0.719, '4': 0.854},  # Table 3
                            'south': {'all':0.400, '1': 0.400, '2': 0.540, '3': 0.713, '4': 0.860},},},   # Table 3
        'pzmax': {'main': {'north': {'all':1.024, '1': 0.545, '2': 0.719, '3': 0.851, '4': 1.024},  # Table 2
                        'south': {'all':1.020, '1': 0.540, '2': 0.713, '3': 0.860, '4': 1.020},},   # Table 2
                'extended': {'north': {'all':1.010, '1': 0.545, '2': 0.719, '3': 0.854, '4': 1.010},  # Table 3
                            'south': {'all':1.000, '1': 0.540, '2': 0.713, '3': 0.860, '4': 1.000},},},   # Table 3
        'logmhalomean': {'1': 13.40, '2': 13.40, '3': 13.24, '4': 13.24},  # p19p1
    }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['pzbin', 'hemisphere', 'sample'])

        # self.info['ngal'] = self.spefdata['ngal'][self.sample]
        self.info['area'] = self.spefinfo['area'][self.sample]
        self.info['ndens'] = self.spefinfo['ndens'][self.sample][self.pzbin]  # [deg^-2]
        if self.pzbin!='all': 
            self.info['zmean'] = self.spefinfo['zmean'][self.sample][self.pzbin]
            self.info['logmhalomean'] = self.spefinfo['logmhalomean'][self.pzbin]
        if self.hemisphere!='combined':
            self.info['pzmin'] = self.spefinfo['pzmin'][self.sample][self.hemisphere][self.pzbin]
            self.info['pzmax'] = self.spefinfo['pzmax'][self.sample][self.hemisphere][self.pzbin]

        zdistfile = f"{self.path}/{self.sample}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"  # open data file
        cols = pd.read_csv(zdistfile, sep=" ", nrows=1).columns[1:]  # get columns from first row
        self.zdfdata = pd.read_csv(zdistfile, sep=" ", skiprows=1, names=cols)  # format into dataframe

        self.z = (self.zdfdata.zmax+self.zdfdata.zmin).values/2  # calculate z bin centers
        pzstr = f'bin_{self.pzbin}' if self.pzbin!='all' else 'all'  # get name of column base on bin
        self.Nz_deg2 = self.zdfdata[f"{pzstr}_{self.hemisphere}"].values  # get raw values from plot
        
        dNdz = self.Nz_deg2*self.info['area']/(self.z[1]-self.z[0])
        self.dNdz = lambda **cosmopars: dNdz


class Gao2023(BaseSMF):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    info = {'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
            'hh':0.71, 'Omega_m':0.268, 'Omega_L':0.732,
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

    def dndlogmstar(self, hh=0.71, **kwargs):  # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
        return self.dndlogmstar_h3*hh**3
    
    def dNdz(self, **cosmopars):
        return np.trapezoid(self.dndlogmstar(**cosmopars), self.logmstar)*self.volumes(**cosmopars)/(self.z[1]-self.z[0])


class SDSSBOSS(BaseSMF):  # (Ahn+ 2013, arxiv.org/abs/1307.7735, Alam+ 2015, https://arxiv.org/abs/1501.00963)    
    galaxys = ['CMASS', 'LOWZ']  # Galaxy sample
    groups = ['portsmouth', 'wisconsin', 'granada']  # Group models
    IMFs = ['Kroupa', 'Salpeter']  # Initial Mass Function
    templates = ['starforming', 'passive']
    pops = ['Bruzual-Charlot', 'Maraston']
    times = ['EarlySF', 'ExtendedSF']
    dusts = ['dust', 'nodust']
    DRs = ['10', '12']
    info = {}
    spefinfo = {'area': {'10':6373.2, '12':9376}}

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['group', 'galaxy', 'DR'])
        self.info['area'] = self.spefinfo['area'][self.DR]

        vers = {'10':'v1_0', '12':'v1_1'}[self.DR]
        self.path = f"/global/cfs/projectdirs/sdss/data/sdss/dr{self.DR}/boss/spectro/redux/galaxy/{vers}"  # Path of the data in NERSC
        if not os.path.isdir(self.path): # Path through a URL if not in NERSC
            self.path = f"https://data.sdss.org/sas/dr{self.DR}/boss/spectro/redux/galaxy/{vers}/"
            
        vers = {'10':'5_12', '12':'7_0'}[self.DR]
        # Each group model needs different specifications and has a different naming scheme
        if self.group=='portsmouth': 
            self.checkspefs(spefs, required=['template', 'IMF'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            fname = f"{self.group}_stellarmass_{self.template}_{imfstr}-v5_{vers}.fits.gz"
        elif self.group=='wisconsin': 
            self.checkspefs(spefs, required=['pop'])
            popstr = {'Bruzual-Charlot':'bc03', 'Maraston':'m11'}[self.pop]
            fname = f"{self.group}_pca_{popstr}-v5_{vers}.fits.gz"
        elif self.group=='granada': 
            self.checkspefs(spefs, required=['IMF', 'time', 'dust'])
            imfstr = {'Kroupa':'krou', 'Salpeter':'salp'}[self.IMF]
            timestr = {'EarlySF':'earlyform', 'ExtendedSF':'wideform'}[self.time]
            fname  = f"{self.group}_fsps_{imfstr}_{timestr}_{self.dust}-v5_{vers}.fits.gz"

        # Fetch the data with properly naming and renaming the mass column
        # https://www.sdss4.org/dr17/spectro/galaxy_portsmouth/
        mcolname = {'portsmouth':'LOGMASS', 'wisconsin':'MSTELLAR_MEDIAN', 'granada':'MSTELLAR_MEDIAN'}[self.group]
        self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

        # Select the correct galaxy sample using the bitmasks in www.sdss3.org/dr10/algorithms/bitmask_boss_target1.php
        # https://www.sdss3.org/dr10/algorithms/bitmasks.php
        # https://www.sdss4.org/dr17/algorithms/bitmasks/
        # https://skyserver.sdss.org/dr19/MoreTools/browser/
        bitmask = {'CMASS':7, 'LOWZ':0}[self.galaxy]
        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (bitmask in bits))]
        
        self.make_SMF()  # Create the distribution from the dataframe

    def make_SMF(self, zbins=None, logmstarbins=None, **kwargs):  # Recreate the distribution from the dataframe with different binning
       self.dndlogmstar, self.dNdz, self.z, self.logmstar = self.bin_catalog(self.dfdata.Z, self.dfdata.LOGM, zbins, logmstarbins)
              
            
      
# # Plot the distributions
# fig, axs = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
# axs[0].plot(halodist.z, halodist.dNdz())
# axs[1].plot(halodist.logmhalo, np.trapezoid(halodist.dndlogmhalo(), halodist.z, axis=0))

# axs[0].set(xlabel=r'$z$', ylabel=r'$\frac{dN}{dz}$')
# axs[1].set(xlabel=r'$\log M_\text{halo}$', ylabel=r'$\frac{dn}{d\log M_\text{halo}}$', yscale='log')
# plt.show()

# # Show info about the distribution based on the paper and our numbers
# ntot0 = np.trapezoid(halodist.dNdz(), halodist.z)
# zave0 = np.trapezoid(halodist.dNdz()*halodist.z, halodist.z)/ntot0
# dndzdlogmhalo_norm0 = halodist.dndlogmhalo()/np.trapezoid(np.trapezoid(halodist.dndlogmhalo(), halodist.logmhalo), halodist.z)
# logmave0 = np.trapezoid(np.trapezoid(halodist.logmhalo*dndzdlogmhalo_norm0, halodist.logmhalo), halodist.z)
# print(f'ntot={ntot0:.2f}, zave={zave0:.2f}, logmave={logmave0:.2f}')
# print(halodist.info)

# # We can add cuts and change the binning with interpolation (or we can use the distribution as is)
# zs = np.linspace(0.4, 0.7, 10)
# logmhalos = np.linspace(12, 14, 50)
# intp_points = np.stack(np.meshgrid(zs, logmhalos, indexing='ij'), axis=-1)
# dndlogmhalo_smf = RegularGridInterpolator((halodist.z, halodist.logmhalo), halodist.dndlogmhalo(), bounds_error=False, fill_value=0)(intp_points)
# dNdz = np.interp(zs, halodist.z, halodist.dNdz())

# # Construct an averaging function from the HMF from data
# dndzdlogmhalo_norm = dndlogmhalo_smf/np.trapezoid(np.trapezoid(dndlogmhalo_smf, logmhalos), zs)
# aveprof_dist = lambda prof: np.trapezoid(np.trapezoid(prof*dndzdlogmhalo_norm, logmhalos), zs)
# logmave = aveprof_dist(logmhalos)
# zave = np.trapezoid(zs*dNdz, zs)/np.trapezoid(dNdz, zs)
# ntot = np.trapezoid(dNdz, zs)
# print(f"ntot={ntot:.2f}, zave={zave:.2f}, logmave={logmave:.2f}")