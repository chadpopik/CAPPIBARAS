"""
Data from studies

"""

import os
from astropy.table import Table
import numpy as np
import astropy
import astropy.cosmology
import astropy.units as u
import astropy.constants as c
import pandas as pd
import h5py, json, tarfile

import Models.Studies as Studies
import Models.HaloModels as HaloModels
import Models.SHMRs as SHMRs


datapath = "/global/homes/c/cpopik/CAPPIBARAS/Data"  # path to data


class BaseData:
    def dNdq_cat(self, qs, dq, qmin=None, qmax=None):
        if qmin is None: qmin = np.floor(qs.min()/dq)*dq  # Min if none are set
        if qmax is None: qmax = np.ceil(qs.max()/dq)*dq  # Max if none are set
        qbins = np.arange(qmin, qmax+dq, dq)  # make bins
        qcents = (qbins[1:]+qbins[:-1])/2  # get center of bins
        dNdq = np.histogram(qs, bins=qbins)[0]/dq /u.dex  # create number dist [dex^-1]
        return dNdq, qcents, qbins
    
    def dNdq1dq2_cat(self, q1s, dq1, q2s, dq2, q1min=None, q1max=None, q2min=None, q2max=None):
        dNdq1, q1cents, q1bins = self.dNdq_cat(q1s, dq1, q1min, q1max)  # Get 1D info for q1
        dNdq2, q2cents, q2bins = self.dNdq_cat(q2s, dq2, q2min, q2max)  # Get 1D info for q2
        N, _, _ = np.histogram2d(q1s, q2s, bins=[q1bins, q2bins])  # Bin catalog into 2D q1/q2 array
        dNdq1dq2 = N /dq1/u.dex /dq2/u.dex  # create number dist [dex^-2]
        return dNdq1dq2, dNdq1, dNdq2, q1cents, q2cents
    


class Jenna_Catalog(BaseData, Studies.Jenna_Catalog):
    path = "/global/homes/c/cpopik/Data/"  # location of data, provided to me by Jenna
    subs = {'masstype': ['Mstar', 'M200c', 'Mvir'], # Mass type (column names)
            }
    
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.dfdata = pd.read_csv(f"{self.path}/ACT_DR6_DESI_Y1Iron_LRGs_valid.csv")  # import datafarme
        
        self.get_dists()

    def get_dists(self, dlogM=0.01, dz=0.05, **kwargs): 
        self.require(['masstype'])
        logMs, zs = np.log10(self.dfdata[self.masstype]), self.dfdata.z
        dNdlogMdz, dNdlogM, self.dNdz, logMs, self.zs = self.dNdq1dq2_cat(logMs, dlogM, zs, dz)
        if self.masstype=='Mstar': self.logMstar, self.dNdlogMstar, self.dNdlogMstar2D = logMs, dNdlogM, dNdlogMdz*dz*u.dex  # if stellar mass, define the arrays as star
        else: self.logMhalo, self.dNdlogMhalo, self.dNdlogMhalo2D = logMs, dNdlogM, dNdlogMdz*dz*u.dex  # if halo mass, define the arrays as halo
        self.dndz = self.dNdz/self.area


class Popik2025(BaseData, Studies.Popik2025):  # TODO: In progress
    path = f"/global/homes/c/cpopik/Stacking_Correlating/Results"
    subs = {
    'zbin': ['z1', 'z2', 'z3', 'z4'],
    'deproj' : ['Base', 'cib', 'cib_cibdBeta', 'cib_cibdBeta_cibdT', 'cib_cibdT'],
    'TCIB': ['10.7', '24.0'],
    'beta': ['1.0', '1.2', '1.4', '1.6', '1.7', '1.8', '2.0'],
    }

    def __init__(self, inputsdict, **inputvars):        
        self.setup(inputsdict | inputvars)
        
        self.require(['deproj'])
        if self.deproj!='cib_cibdBeta_cibdT': # Add values
            self.subs['TCIB'] = ['10.7']
        if self.deproj!='cib_cibdBeta': # Add values
            self.subs['TCIB'] = self.subs['TCIB']+['1.15', '1.30', '1.35', '1.45']

        if self.deproj=='Base': self.require(['deproj'])
        else: self.require(['deproj', 'TCIB', 'beta'])

        ACTDR6 = Coulton2024()
        self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data

        self.get_meas()

        # Taking properties from Zhou
        Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
        for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logmhalomean', 'zmean']:
            setattr(self, val, getattr(Zhou, val))

    def get_meas(self):
        with h5py.File(f"{self.path}/ACTDR6DESILRG_Spectra_testnew.h5", 'r') as f:
            self.ell = f['ell'][()]
            self.Cgg_data = f[f'gxg/{self.zbin[-1]}'][()]
            self.Cgy_data = f[f'gxy/{self.zbin[-1]}/{self.deproj}/{self.TCIB}/{self.beta}'][()]
            self.Cyy_data = f[f'yxy/{self.deproj}/{self.TCIB}/{self.beta}'][()]

            self.Cgy_err = np.abs(self.Cgy_data)/10000*self.ell  # TODO: placeholder

class RiedGuachalla2025(BaseData, Studies.RiedGuachalla2025):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs (arxiv.org/abs/2503.19870)
    path = f"{datapath}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass bin
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['bin'])

        ACTDR6 = Coulton2024()
        self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data

        self.get_meas()  # get the measurements
        self.get_dNdz()  # get the redshift distribution
        self.get_dNdlogmstar()  # get the mass distribution
        
    def get_meas(self):
        # TODO: Add more of the plots as options
        
        TkSZ_fid = dict(np.load(f"{self.path}/fig8_fiducial.npz")) # The measured stacked kSZ in μK arcmin2 for varying CAP filters with radius R
        TkSZ_zbins = dict(np.load(f"{self.path}/fig11_ksz_z.npz"))  # Mean stacked kSZ profiles for the different redshift bins
        TkSZ_mbins = dict(np.load(f"{self.path}/fig12_ksz_mass.npz"))  # kSZ stacked profiles for the different stellar mass bins, denoted by mass

        if self.bin=='all':
            self.R, self.TkSZ_data, self.TkSZ_err = TkSZ_fid['R'] *u.arcmin, TkSZ_fid['DESIxACT'] *u.uK*u.arcmin**2, TkSZ_fid['errors_DESIxACT'] *u.uK*u.arcmin**2

        elif self.bin[0]=='z':
            self.R, self.TkSZ_data, self.TkSZ_err = TkSZ_zbins['R'] *u.arcmin, TkSZ_zbins[f'{self.bin}'] *u.uK*u.arcmin**2, TkSZ_zbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        elif self.bin[0]=='m':
            self.R, self.TkSZ_data, self.TkSZ_err = TkSZ_mbins['R'] *u.arcmin, TkSZ_mbins[f'{self.bin}'] *u.uK*u.arcmin**2, TkSZ_mbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

    def get_dNdz(self, dz = 0.01):  # makes the redshift distribution with arbritrary binsize dz
        self.allcat_zs = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # Redshifts of all galaxies
        
        if self.bin[0]=='z': self.cat_zs = self.allcat_zs[f'{self.bin}']  # if it's for a z bin, only get a subset of zs
        else: self.cat_zs = np.concatenate([self.allcat_zs[f'z_{i}'] for i in range(1, 5)])  # otherwise get all of them
            
        self.dNdz, self.zs, zbins = self.dNdq_cat(self.cat_zs, dz)  # Get 1D hist for z
        self.dndz = self.dNdz/(self.area*u.deg**2) # create number dens dist [deg^-2 dex^-1]

    def get_dNdlogmstar(self, dlogmstar=0.05):
        self.allcat_mstar = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # Stellar masses of all galaxies
        
        if self.bin[0]=='m': self.cat_ms = self.allcat_mstar[f'{self.bin}']  # if it's for a m bin, only get a subset of ms
        else: self.cat_ms = np.concatenate([self.allcat_mstar[f'mass_{i}'] for i in range(1, 5)])  # otherwise get all of them
                    
        self.dNdlogMstar, self.logMstar, Mstarbins = self.dNdq_cat(np.log10(self.cat_ms), dlogmstar)  # Get 1D hist for z

    # def get_dndlogmhalo(self, dlogmhalo=0.05):  # TODO: This part requires some outside info and more work
    #     self.get_dNdlogmstar()
        
    #     shmr = lambda logmstar: SHMRs.Gao2023({'sample':'Psat_Mh'}).SHMR(logmstar)()  # steal default SHMR from Gao 2023
    #     logmhalobins = np.arange(shmr(self.logmstarmin), shmr(self.logmstarmax)+dlogmhalo, dlogmhalo)  # make logm bins
    #     self.logmhalos = (logmhalobins[1:]+logmhalobins[:-1])/2  # get (log) center of bins
        
    #     self.dNdlogmhalo = np.histogram(shmr(np.log10(self.cat_ms)), bins=logmhalobins)[0]/dlogmhalo  # create number dist [dex^-1]
        
    #     # norm_fac = self.dNdz[:,None]/np.trapz(self.dNdlogmhalo, self.logmhalos)*(self.zs[1]-self.zs[0])  # ensures same zcount as dNdz
    #     # self.dndlogmhalo = self.dNdlogmhalo*norm_fac/self.volumes()[:, None]  # create number dist [dex^-1]



class Hadzhiyska2025(BaseData, Studies.Hadzhiyska2025):  # Stacked kSZ measurement of ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
    path = f"{datapath}/Hadzhiyska2024"  # Path to data from zenodo.org/records/12633573
    subs = {
        'zbin': ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'DR':['all', 'DR9', 'DR10'],
        'sample': ['main', 'extended', 'all'],
        'zoutcut': ['nocut', 'cut'],
        'corr': ['corrected', 'uncorrected'],
    }
    
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'DR', 'sample', 'zoutcut', 'corr'])

        ACTDR6 = Coulton2024()
        self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data

        simsload = np.load(f"{self.path}/Fig2_sim.npz")  # Stacked kSZ signal of Main sample, all z bins, and from TNG and Illustris models
        self.R = simsload['theta_arcmins'] *u.arcmin
        self.Tksz_Illustris1 = simsload['gas_illustris'] *u.uK*u.arcmin**2
        self.Tksz_TNG300 = simsload['dm_tng']  *u.uK*u.arcmin**2
        self.Tksz = simsload['signal']  *u.uK*u.arcmin**2

        samplestr = {'main': '', 'extended': 'extended_', 'all': ''}[self.sample]
        corrstr = {'corrected':'corr', 'uncorrected':''}[self.corr]
        zstr = {'nocut': '', 'cut': 'sigmaz0.05000_'}[self.zoutcut]
        filename = f"{self.path}/Fig1_Fig8_{samplestr}dr10_allfoot_perbin_{zstr}dr6_{corrstr}pzbin{self.zbin[-1]}.npz"

        self.TkSZ_data = np.load(filename)['prof'] *u.uK*u.arcmin**2
        self.TkSZ_cov = np.load(filename)['cov'] *(u.uK*u.arcmin**2)**2
        self.TkSZ_err = np.diag(self.TkSZ_cov)**0.5


class Liu2025(BaseData, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs (Liu+ 2025, arxiv.org/abs/2502.08850)
    path = f"{datapath}/Liu2025"  # path to data from zenodo.org/records/14706729
    sharedpath = f"/global/homes/c/cpopik/Data/Liu2025_shared"  # path to data from author
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        'dp' :['beta', 'dBeta'],  # CIB deprojection method (DR6)
        'Beta' : ['fiducial', '1.2', '1.4', '1.6'],  # spectral index value in CIB deprojection (DR6)
        'TCIB' : ['10.7', '24.0'],  # TCIB value in CIB dprojection (DR6)
        'aper' : ['CAP', 'RingRing'],  # aperture used (DR5)
        'freq' : ['090', '150'],  # y map frequency (DR5)
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'ACTDR'])

        if self.ACTDR=='DR6': 
            self.require(['zbin', 'dp', 'Beta', 'TCIB'])
            ACTDR6 = Coulton2024()
            self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data
        elif self.ACTDR=='DR5': 
            self.require(['zbin', 'aper', 'freq'])
            ACTDR5 = Naess2020({'freq':self.freq})
            self.beam_ells, self.beam_data = ACTDR5.beam_ells, ACTDR5.beam_data
            self.resp_ells, self.resp_data = ACTDR5.resp_ells, ACTDR5.resp_data

        self.get_meas()
        self.get_dNdz()
        
        try: self.get_meas_shared()
        except: pass

    def get_meas(self):
        if self.ACTDR=='DR6':  # ACT DR6 stacked y profiles, fiducial and using deprojection method/values:
            if self.dp=='beta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig3.csv")  # using CIB deprojected y maps
            elif self.dp=='dBeta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig4.csv")  # using CIB & dBeta moment deprojected y maps
            elif self.dp=='beta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig10.csv")  # using CIB deprojected y maps, T=24.0K
            elif self.dp=='dBeta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig11.csv")  # using CIB & dBeta moment deprojected y maps, T=24.0K
            
            dBetastr = f"Beta_{self.Beta}" if self.Beta!='fiducial' else 'fiducial'
            
            self.R = y['RApArcmin'].values[:-1] *u.arcmin
            self.y_data= y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}"].values[:-1] *u.arcmin**2
            self.y_err = y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}_err"].values[:-1] *u.arcmin**2
            
        elif self.ACTDR=='DR5':  # ACT DR5 stacked y profiles
            if self.aper=='CAP': y = pd.read_csv(f"{self.path}/fig12.csv")  # using standard CAP
            elif self.aper=='RingRing': y = pd.read_csv(f"{self.path}/fig13.csv")  # using ring-ring filter
            
            self.R = y['RApArcmin'].values *u.arcmin
            self.y_data = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}"].values *u.arcmin**2
            self.y_err = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}_err"].values *u.arcmin**2
            
    def get_meas_shared(self):
        if self.ACTDR=='DR6' and self.TCIB=='10.7':
            dBetastr = f"dBeta_{self.Beta}_{self.TCIB}" if self.Beta!='fiducial' else 'fiducial'
            y = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/diskring_tsz_uniformweight_measured.txt").T
            ycov = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/cov_diskring_tsz_uniformweight_bootstrap.txt").T

            self.R = y[0] *u.arcmin
            self.y_data = (y[1] *u.sr).to(u.arcmin**2)
            self.y_err = (y[2] *u.sr).to(u.arcmin**2)
            self.y_cov = (ycov *u.sr).to(u.arcmin**2)**2

    def get_dNdz(self):
        # Loading redshift distribution from file
        zcols = pd.read_csv(f"{datapath}/Zhou2023B/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # col names from Zhou2023
        zdf = pd.DataFrame(np.genfromtxt(f"{self.path}/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"), columns=zcols)  # Spectroscopic distributions of four sub-sample photometric redshift bins
        self.zs = (zdf.zmin+zdf.zmax).values/2  # define zs at center of bins
        self.dz = self.zs[1]-self.zs[0]
        self.dndz = zdf[f'bin_{self.zbin[-1]}_combined'].values/self.dz /u.deg**2/u.dex
        self.dNdz = self.dndz*(self.area*u.deg**2)
        
        
class Coulton2024(BaseData, Studies.Coulton2024):  # ACT DR6 ILC maps (Coulton 2024, arxiv.org/abs/2307.01258)
    path = f"{datapath}/ACTDR6"  # Path to data downloaded from portal.nersc.gov/project/act/dr6_nilc/ymaps_20230220/
    # NERSC_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220"  # path to data in NERSC
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        ybeamdf = pd.read_csv(f"{self.path}/ilc_beam.txt", sep=" ")
        self.beam_ells, self.beam_data = ybeamdf['#'].values, ybeamdf['ell'].values



class Gao2023(BaseData, Studies.Gao2023):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    path = f"{datapath}/Gao2023"
    subs = {'sample':['LRG', 'ELG']}  # Galaxy Sample

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])

        self.get_dndlogmstar()

    def get_dndlogmstar(self):
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.zs = (zbins[1:]+zbins[:-1])/2
        
        # Read the plot data from the files
        self.logMstars = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        dndlogMstar_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]
        self.dndlogMstar = dndlogMstar_h3/u.Mpc**3*self.h**3/u.dex
        

    # def dndlogmstar(self, hh=0.71, **kwargs):  # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    #     return self.dndlogmstar_h3*hh**3
    
    # def dNdz(self, **cosmopars):
    #     return np.trapz(self.dndlogmstar(**cosmopars), self.logmstar)*self.volumes(**cosmopars)/(self.z[1]-self.z[0])
    
    
class Kou2023(BaseData, Studies.Kou2023):
    path = "/global/homes/c/cpopik/Data/Kou2023"  # path to data, taken from plots using webplotdigitizer
    subs = {'mbin':['M1', "M2", "M3", "M4"]}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['mbin'])
            
        self.get_meas()

    def get_meas(self):
        with h5py.File(f'{self.path}/Kou2023_wpd.h5', "r") as f:
            self.Cgg_ell = f['dgg_ells'][()]
            self.Cgy_ell = f['dgy_ells'][()]
            self.Cgg_data = f[f'dgg_{self.mbin}'][()]/(self.Cgg_ell*(self.Cgg_ell+1))*2*np.pi
            self.Cgy_data = f[f'dgy_{self.mbin}'][()]/(self.Cgy_ell*(self.Cgy_ell+1))*2*np.pi


class Zhou2023(BaseData, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B"  # path to data from zenodo.org/records/8319955
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }

    def __init__(self, inputsdict, **inputvars):       
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'sample', 'hemisphere'])

        self.get_meas()
        self.get_dndz()
        
    def get_meas(self):
        samp = {'main':'fid', 'ext':'ext'}[self.sample]

        Cls = json.load(open(f"{self.path}/combined_cls.json", "r"))
        self.ell = np.array(Cls['ell'])
        self.Cgg_data = np.array(Cls[f'cls_{samp}'][f's0{self.zbin[-1]}'])
        
        self.Wggs = json.load(open(f"{self.path}/combined_wth.json", "r"))
        self.thetas = np.array(self.Wggs['theta(deg)'])  # [deg]
        self.Wgg_data = np.array(self.Wggs[f'wth_{samp}'][f's0{self.zbin[-1]}'])  # angular correlation function

    def get_dndz(self):
        samp = {'main':'main', 'ext':'extended'}[self.sample]
        cols = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get columns from first row
        zdf = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", skiprows=1, names=cols)  # format into dataframe
        self.zs = (zdf.zmin.values+zdf.zmax.values)/2
        pzstr = f'bin_{self.zbin[-1]}_{self.hemisphere}' if self.zbin!='all' else 'all'  # get name of column corresponding on bin
        self.dz = self.zs[1]-self.zs[0]
        self.dndz = zdf[pzstr].values//self.dz /u.deg**2/u.dex
        self.dNdz = self.dndz * (self.area*u.deg**2)


class Kusiak2022(BaseData, Studies.Kusiak2022):  # HOD for unWISE galaxies and Planck lensing arxiv.org/abs/2203.12583
    path = "/global/homes/c/cpopik/Data/Kusiak2022"  # path to data, taken from plots using webplotdigitizer
    subs = {'sample':['Blue', 'Green', 'Red']}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])
        
        self.get_meas()
        
    def get_meas(self):
        with h5py.File(f'{self.path}/Kusiak2022_wpd.h5', "r") as f:
            self.Cgg_data = f[f'{self.sample}'][()]/1e5
            self.ell = f['ell'][()]


class White2022(BaseData, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing (White+ 2022, arxiv.org/abs/2111.09898)
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

        self.require(['zbin'])
            
        self.get_meas()
        self.get_dNdz()

    def get_meas(self):
        self.ell, self.Cgg_data = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cls.txt").T[0:2]  # ells and measured angular auto-spectra
        self.ell_model, self.Cgg_data_model = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_mod.txt").T[0:2]  # smooth model used to calculate cov mat
        
        self.C_covcomb = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cov.txt")  # combined covariance matrix of auto and cross
        self.Cgg_cov = self.C_covcomb[:int(self.ell.size)]  # TODO: lazely assuming i can do this
        self.Cgg_err = np.diag(self.Cgg_cov)**0.5
        
        # weights of all multipoles used to get effective multipole ell, ell_effective = np.sum(weights*ells)
        self.auto_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wla.txt")  # window function matrix for auto-spec
        self.cross_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wlx.txt")  # window function matrix for cross-spec
        
    def get_dNdz(self):
        self.zs, self.nz = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_dndz.txt").T  # normalized redshift distribution [deg^{-2} dex^-1]
        self.dz = self.zs[1]-self.zs[0]
        self.dndz = self.nz/self.dz /u.deg**2/u.dex
        self.dNdz = self.dndz*self.area


class Amodeo2021(BaseData, Studies.Amodeo2021):  # Inference on BOSS DR10 stacked ACT DR5 (arxiv.org/abs/2009.05558)
    path = '/global/homes/c/cpopik/Data/Amodeo2021/'  # path to data, taken from plots using webplotdigitizer and various repos
    subs = {'prof': ['Amodeo', 'Battaglia', 'TNG'],
                'units': ['cosmo', 'cgs']}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['model', 'units'])

        self.get_meas()
        
    def get_meas(self):        
        with h5py.File(f"{self.path}/Amodeo2021_wpd.h5", "r") as f:
            self.rs_rho = f[f'rs_rho_{self.prof}'][()]
            self.rs_pth = f[f'rs_pth_{self.prof}'][()]
            self.rho = f[f'rho_{self.prof}_{self.units}'][()]
            self.pth = f[f'pth_{self.prof}_{self.units}'][()]
        
        cmass2h = np.loadtxt(f'{self.path}/twohalo_cmass_average.txt').T
        self.rs_2hCMASS = cmass2h[0]*u.Mpc/0.7  # [Mpc/h] -> [Mpc]
        self.rho_2hCMASS = cmass2h[1]*u.g/u.cm**3
        self.pth_2hCMASS = cmass2h[2]*u.g/u.cm/u.s**2
        
        if self.units=='cosmo':
            self.rho = (self.rho *u.Msun/u.kpc**3).to(u.Msun/u.Mpc**3)
            self.pth = (self.pth *u.Msun/u.kpc/u.s**2).to(u.Msun/u.Mpc/u.s**2)
            self.rho_2hCMASS = self.rho_2hCMASS.to(u.Msun/u.Mpc**3)
            self.pth_2hCMASS = self.pth_2hCMASS.to(u.Msun/u.Mpc/u.s**2)

        elif self.units=='cgs':
            self.rho = self.rho *u.g/u.cm**3
            self.pth = self.pth *u.g/u.cm/u.s**2
        
        self.get_dNdlogmhalo()
        
    def get_dNdlogmhalo(self, dlogmhalo=0.05):
        massdist = np.loadtxt(f'{self.path}/mass_distrib.txt')        
        self.dNdlogMhalo, self.logMhalo, Mhalobins = self.dNdq_cat(np.log10(massdist), dlogmhalo)  # Get 1D hist for z

        

class Schaan2021(BaseData, Studies.Schaan2021):  # ACT DR5 maps stacked on SDSS BOSS DR10 (Schaan+ 2021, arxiv.org/abs/2009.05557)
    path = f"{datapath}/Schaan2021"  # path to data, shared by author
    subs = {
        'sample': ['cmass', 'lowz'],
        'freq' : ['150', '090'],  # frequency band of obsevation [GHz]
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

        self.require(['sample', 'freq'])
           
        ACTDR5 = Naess2020({'freq':self.freq})
        self.beam_ells, self.beam_data = ACTDR5.beam_ells, ACTDR5.beam_data
        self.resp_ells, self.resp_data = ACTDR5.resp_ells, ACTDR5.resp_data

        self.get_meas()  # get measurement
        
    def get_meas(self):
        measpath = f"{self.path}/{self.sample}_data_sharing_schaan21/f{self.freq}"  # each meas in different folder
        if self.sample=='cmass':
            self.R = np.genfromtxt(f"{measpath}/diskring_tsz_varweight_measured.txt").T[0] *u.arcmin
            self.TkSZ_data, self.TkSZ_err = (np.genfromtxt(f"{measpath}/diskring_ksz_varweight_measured.txt").T[1:] *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.TtSZ_data, self.TtSZ_err = (np.genfromtxt(f"{measpath}/diskring_tsz_varweight_measured.txt").T[1:] *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.TkSZ_cov = (np.genfromtxt(f"{measpath}/cov_diskring_ksz_varweight_bootstrap.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)
            self.TtSZ_cov = (np.genfromtxt(f"{measpath}/cov_diskring_tsz_varweight_bootstrap.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)
    
        elif self.sample=='lowz':
            freqstr = str(int(self.freq))
            self.R = np.genfromtxt(f"{measpath}/ksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T[0] *u.arcmin
            self.TkSZ_data = (np.genfromtxt(f"{measpath}/ksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt") *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.TtSZ_data = (np.genfromtxt(f"{measpath}/tsz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt") *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.TkSZ_cov = (np.genfromtxt(f"{measpath}/covksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)
            self.TtSZ_cov = (np.genfromtxt(f"{measpath}/covtsz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)

        # Convert to y units
        
        convfac = 1/HaloModels.y_to_uK(np.float32(self.freq)*u.GHz, self.T_CMB*u.K)
        self.y_data = self.TtSZ_data*convfac
        self.y_cov = self.TtSZ_cov*convfac**2

        for val in ['TkSZ', 'TtSZ', 'y']:  # get errors from covariance matrices
            setattr(self, f'{val}_err', np.diag(getattr(self, f'{val}_cov'))**0.5) 


class Naess2020(BaseData, Studies.Naess2020):  # ACT DR5 (Naess 2020, arxiv.org/abs/2007.07290)
    path = f"{datapath}/ACTDR5"  # Path to data downloaded from lambda.gsfc.nasa.gov/product/act/actpol_dr5_aux_prod_get.html
    # NERSC_path = "/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary"  # location of data in NERSC
    
    subs = {'freq': ['090', '150', '220']}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['freq'])

        self.beamfile = f"{self.path}/beams/act_planck_dr5.01_s08s18_f{self.freq}_daynight_beam.txt"  # Map beams transfer function: ells, B
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]

        self.respfile = f"{self.path}/responses/act_planck_dr5.01_s08s18_AA_f{self.freq}_daynight_response_tsz.txt"  # Map-averaged response to tSZ: ell, I, dI, Q, dQ, U, dU
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]  # [ells, uk/y]



class Koukoufilippas2020(BaseData, Studies.Koukoufilippas2020):
    path = "/global/homes/c/cpopik/Data/Koukoufilippas2020"  # path to data, taken from plots using webplotdigitizer
    subs = {'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5']}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])

        self.get_meas()

    def get_meas(self):
        self.Cgy_data = self.Cgy[self.sample]
        self.Cgy_ell = self.ells[self.sample]

    def get_meas(self):        
        with h5py.File(f'{self.path}/Koukoufilippas2020_wpd.h5', "r") as f:
            self.Cgy_ell = f[f'ell_{self.sample}'][()]
            self.Cgy_data = f[f'Cgy_{self.sample}'][()]
            

class SDSSBOSS(BaseData, Studies.Ahn2013Alam2015):  # (Ahn+ 2013, arxiv.org/abs/1307.7735, Alam+ 2015, https://arxiv.org/abs/1501.00963)
    path = '/global/cfs/projectdirs/sdss/data/sdss'   # Path of the data in NERSC
    if not os.path.isdir(path): # Path through a URL if not in NERSC
        path = 'https://data.sdss.org/sas/'
    subs = {
        'DR': ['DR10', 'DR12'],
        'galaxy': ['CMASS', 'LOWZ'],  # Galaxy sample
        'group': ['portsmouth', 'wisconsin', 'granada'],  # Group models
        'IMF': ['krou', 'salp'],  # Kroupe or Salpeter Initial Mass Function
        'template': ['starforming', 'passive'],
        'pop': ['bc03', 'm11'], # Bruzual-Charlot or Maraston population
        'time': ['earlyform', 'wideform'],  # early or extended SF
        'dust': ['dust', 'nodust'],
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['DR'])
        
        self.get_catalog()
        self.get_dists()   
        
    def get_catalog(self):
        self.require(['group', 'galaxy'])
        vers = {'DR10':['v1_0', '5_12'], 'DR12':['v1_1', '7_0']}[self.DR]
        self.path = f"{self.path}/dr{self.DR[2:]}/boss/spectro/redux/galaxy/{vers[0]}"
            
        if self.group=='portsmouth': 
            self.require(['template', 'IMF'])
            fname = f"{self.group}_stellarmass_{self.template}_{self.IMF}-v5_{vers[1]}.fits.gz"
        elif self.group=='wisconsin': 
            self.require(['pop'])
            fname = f"{self.group}_pca_{self.pop}-v5_{vers[1]}.fits.gz"
        elif self.group=='granada': 
            self.require(['template', 'time', 'dust'])
            fname = f"{self.group}_fsps_{self.IMF}_{self.time}_{self.dust}-v5_{vers[1]}.fits.gz"

        # Fetch the data with properly naming and renaming the mass column, sdss4.org/dr17/spectro/galaxy_portsmouth/
        mcolname = {'portsmouth':'LOGMASS', 'wisconsin':'MSTELLAR_MEDIAN', 'granada':'MSTELLAR_MEDIAN'}[self.group]
        self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

        # Select the correct galaxy sample using the bitmasks in sdss3.org/dr10/algorithms/bitmask_boss_target1.php, sdss3.org/dr10/algorithms/bitmasks.php, sdss4.org/dr17/algorithms/bitmasks/, skyserver.sdss.org/dr19/MoreTools/browser/
        bitmask = {'CMASS':7, 'LOWZ':0}[self.galaxy]
        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (bitmask in bits))]
        
    def get_dists(self, dlogMstar=0.05, dz=0.05):
        self.dNdlogMstardz, self.dNdlogMstar, self.dNdz, self.logMstar, self.zs = self.dNdq1dq2_cat(self.dfdata.LOGM, dlogMstar, self.dfdata.Z, dz)
        self.dNdlogMstar2D = self.dNdlogMstardz*dz*u.dex
        self.dndz = self.dNdz/self.area
        

class CAMELShalo(BaseData):  # Random halo
    path = "/global/homes/c/cpopik/Data/CAMELS"
    subs = {'units':['cgs','cosmo']}

    def __init__(self, inputs, **kwargs):
        self.units = inputs['units']
        self.get_meas()
        
    def get_meas(self):
        with h5py.File(f"{self.path}/CAMELS_halo.h5", "r") as f:
            self.rs = f['rs'][()] *u.Mpc
            self.rho = f['rho'][()] *u.g/u.cm**3
            self.pth = f['pth'][()] *u.g/u.cm/u.s**2

        if self.units=='cosmo':
            self.rho = self.rho.to(u.Msun/u.Mpc**3)
            self.pth = self.pth.to(u.Msun/u.Mpc/u.s**2)
            
            
# class Sailer2025:  # TODO: In progress DESI LS LRGs correlated with Planck PR4&ACT DR6 Lensing (Sailer+ 2025, arxiv.org/abs/2407.04607)
#     path = f"{datapath}/Sailer2025"  # Path to data from zenodo.org/records/12613408


# class Qu2025:  # TODO: In progress DESI LS Galaxies correlated with Planck PR4&ACT DR6 Lensing (Qu+ 2025, arxiv.org/abs/2410.10808)
#     path = f"{datapath}/Qu2025"  # Path to data from zenodo.org/records/13844390
    
# class Maus2025:  # TODO: In progress DESI DR1 Galaxies correlated with Planck PR4&ACT DR6 Lensing (Maus+ 2025, arxiv.org/abs/2505.20656)
#     path = f"{datapath}/Maus2025"  # Path to data from zenodo.org/records/17636841