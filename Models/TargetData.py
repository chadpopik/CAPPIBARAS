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
from scipy.interpolate import RegularGridInterpolator

import Models.Studies as Studies
import Models.HaloModels as HaloModels
import Models.SHMRs as SHMRs



datapath = "/global/homes/c/cpopik/Data"  # path to data


        # TODO: handle this stuff
        # norm_fac = self.dNdz[:,None]/np.trapz(self.dNdlogmhalo, self.logmhalos)*(self.zs[1]-self.zs[0])  # ensures same zcount as dNdz
        # self.dndlogmhalo = self.dNdlogmhalo*norm_fac/self.volumes()[:, None]  # create number dist [dex^-1]

class BaseTargetData:
    def make_N_q(self, qs, dq, qmin=None, qmax=None):  # make dist from samples with one value
        if qmin is None: qmin = np.floor(qs.min()/dq)*dq  # default min
        if qmax is None: qmax = np.ceil(qs.max()/dq)*dq  # default max
        qbins = np.arange(qmin, qmax+dq, dq)  # make bins
        qcents = (qbins[1:]+qbins[:-1])/2  # get center of bins
        N_q = np.histogram(qs, bins=qbins)[0]  # make number dist from hist
        # dNdq = N_q /dq/u.dex  # apply units [dex^-1]
        return N_q, qcents, qbins

    def make_N_q1_q2(self, q1s, dq1, q2s, dq2, q1min=None, q1max=None, q2min=None, q2max=None):  # make 2D dist from samples with two values
        N_q1, q1cents, q1bins = self.make_N_q(q1s, dq1, q1min, q1max)  # make 1D dist of value 1
        N_q2, q2cents, q2bins = self.make_N_q(q2s, dq2, q2min, q2max)  # make 1D dist of value 1
        N_q1_q2, _, _ = np.histogram2d(q1s, q2s, bins=[q1bins, q2bins])  # make 2D number dist from hist
        # dNdq1dq2 = N /dq1/u.dex /dq2/u.dex  # apply units [dex^-2]
        return N_q1_q2, N_q1, N_q2, q1cents, q2cents

    # def get_dist(self, dist, dz=None, zMin=None, zMax=None,
    #              dlogMs=None, logMsMin=None, logMsMax=None):  # get redshift distribution with given cuts
    #     if 'z' in dist:
    #         self.make_zdists(dz=dz, zMin=zMin, zMax=zMax)
    #         if dist=='dNdz': dist=self.dNdz
    #         elif dist=='dndz': dist=self.dndz
    #         elif dist=='n_z': dist=self.n_z
    #         elif dist=='N_z': dist=self.N_z
    #         else: raise ValueError("Give valid zdist: dNdz, dndz, n_z, N_z")
    #         return self.z, dist
    #     elif 'Ms' in dist: 
    #         self.make_Msdists(dlogMs=dlogMs, logMsMin=logMsMin, logMsMax=logMsMax)
    #         if dist=='dNdlogMs': dist=self.dNdlogMs
    #         elif dist=='dndlogMs': dist=self.dndlogMs
    #         elif dist=='n_logMs': dist=self.n_logMs
    #         elif dist=='N_logMs': dist=self.N_logMs
    #         else: raise ValueError("Give valid zdist: dNdlogMs, dndlogMs, n_logMs, N_logMs")
    #         return self.logMs, dist
        # elif
        #     self.make_zMsdists(dz=None, zMin=None, zMax=None, dlogMs=dlogMs, logMsMin=logMsMin, logMsMax=logMsMax)
        #     if dist=='dNdlogMs': dist=self.dNdlogMs
        #     elif dist=='dndlogMs': dist=self.dndlogMs
        #     elif dist=='n_logMs': dist=self.n_logMs
        #     elif dist=='N_logMs': dist=self.N_logMs
        #     else: raise ValueError("Give valid zdist: dNdlogMs, dndlogMs, n_logMs, N_logMs")
        #     return self.logMs, dist

    # def get_mhdist(self, mdist, dlogMh=None, logMhMin=None, logMhMax=None):  # get redshift distribution with given cuts
    #     self.make_zdists(dz=dz, zMin=logMhMin, zMax=logMhMax)
    #     cut = (self.z>=(-np.inf if logMhMin is None else logMhMin)) & \
    #           (self.z<=(np.inf if logMhMax is None else logMhMax))
    #     if zdist=='dNdz': zdist=self.dNdz
    #     elif zdist=='dndz': zdist=self.dndz
    #     elif zdist=='n': zdist=self.n
    #     elif zdist=='N': zdist=self.N
    #     else: raise ValueError("Give valid zdist: dNdz, dndz, n, N")
    #     return self.z[cut], zdist[cut]

    # def cut_dNdlogMh(self, logMhMin, logMhMax):
    #     cut = (self.logMhs>=logMhMin) & (self.logMhs<=logMhMax)
    #     return self.logMhs[cut], self.dNdlogMh[cut]

    # def cut_dNdlogMh2D(self, zMin, zMax, logMhMin, logMhMax):
    #     mcut = (self.logMhs>=logMhMin) & (self.logMhs<=logMhMax)
    #     zcut = (self.zs>=zMin) & (self.zs<=zMax)
    #     return self.zs[zcut], self.logMhs[mcut], self.dNdlogMh2D[zcut, mcut]
    
    

class Kusiak2022(BaseTargetData, Studies.Kusiak2022):  # unWISE galaxies and Planck lensing
    path = "/global/homes/c/cpopik/Data/Kusiak2022"  # path to data, provided by author
    subs = {'sample':['Blue', 'Green', 'Red']}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def make_zdists(self, dz=None, zMin=None, zMax=None):
        self.require(['sample'])
        # load from digitized data
        sampstr = {'Blue':0,'Green':1,'Red':2}[self.sample]
        self.zs_df, dNdz_norm = np.loadtxt(f"{self.path}/normalised_dndz_cosmos_{sampstr}.txt").T  # normalized differential number count
        
        zmin = zMin if zMin is not None else self.zs_df.min()
        zmax = zMax if zMax is not None else self.zs_df.max()
        self.dz = dz if dz is not None else self.zs_df[1]-self.zs_df[0]
        self.z = np.arange(zmin, 
                           zmax+self.dz, self.dz)
        self.dNdz_norm = np.interp(self.z, self.zs_df, dNdz_norm)  # interpolate to desired z
        
        # TODO: Need number of galaxies or galaxy density to get unnormed values
        
        
class White2022(BaseTargetData, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing (White+ 2022, arxiv.org/abs/2111.09898)
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
            
    def make_zdists(self, zbin=None, dz=None, zMin=None, zMax=None):
        self.require(['zbin'])
        self.z_df, self.nz_df = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_dndz.txt").T  # density distribution [deg^{-2}]
        
        zmin = zMin if zMin is not None else self.z_df.min()
        zmax = zMax if zMax is not None else self.z_df.max()
        self.dz = dz if dz is not None else self.z_df[1]-self.z_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)
        self.n_z = np.interp(self.z, self.z_df, self.nz_df) /u.deg**2
        self.N_z = self.n_z*self.area
        self.dNdz = self.N_z/self.dz /u.dex
        self.dndz = self.n_z/self.dz /u.dex
        self.dlnNdz = np.log(self.N_z.value)/self.dz/u.dex
        
        # TODO: think these units of density are weird, check



class RiedGuachalla2025(BaseTargetData, Studies.RiedGuachalla2025):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs
    path = f"{datapath}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass subsample
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def make_zdists(self, dz=None, zMin=None, zMax=None, **kwargs):
        # Load redshifts from file
        self.require(['bin'])
        self.allcat_zs = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # redshifts of all galaxies in catalog
        if self.bin[0]=='z': self.cat_zs = self.allcat_zs[f'{self.bin}']  # if zbin is specified, select for that subsample
        else: self.cat_zs = np.concatenate([self.allcat_zs[f'z_{i}'] for i in range(1, 5)])  # otherwise, get all of them

        # Bin redshifts into histogram of number distribution [unitless]
        self.N_z, self.z, self.zbins = self.make_N_q(self.cat_zs, 
                                                dq=dz if dz is not None else 0.01,  # arbitrary default dz
                                                qmin=zMin if zMin is not None else self.zMin,  # default min z of (sub)sample from Studies
                                                qmax=zMax if zMax is not None else self.zMax)  # default max z of (sub)sample from Studies
        self.dz = self.z[1]-self.z[0]  # width of z bins
        self.dNdz = self.N_z/self.dz  # differential number distribution [dex^-1]
        self.n_z = self.N_z/(self.area*u.deg**2)  # number density distribution [deg^-2 dex^-1]
        self.dndz = self.n_z/self.dz  # differential number density distribution [deg^-2 dex^-1]

    def make_Msdists(self, dlogMs=None, logMsMin=None, logMsMax=None, HSMR=None):
        # Load stellar masses from file
        self.require(['bin'])
        self.allcat_mstar = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # stellar masses of all galaxies
        if self.bin[0]=='m': self.cat_ms = self.allcat_mstar[f'{self.bin}']  # if zbin is specified, select for that subsample
        else: self.cat_ms = np.concatenate([self.allcat_mstar[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
    
        # Bin stellar masses into histogram of number distribution [unitless]
        self.N_logMs, self.logMs, self.logMsbins = self.make_N_q(np.log10(self.cat_ms),
                                                            dq=dlogMs if dlogMs is not None else 0.01,  # arbitrary default Ms
                                                            qmin=logMsMin if logMsMin is not None else self.logMsMin,  # default min Ms of (sub)sample from Studies
                                                            qmax=logMsMax if logMsMax is not None else self.logMsMax)  # default max Ms of (sub)sample from Studies
        self.dlogMs = self.logMs[1]-self.logMs[0]  # width of logMs bins
        self.dNdlogMs = self.N_logMs/self.dlogMs  # differential number distribution [Msol^-1]

        # TODO: add number density distribution [Msol^-1 Mpc^-3] (need volume info from Studies)

    def make_Mhdists(self, dlogMh=None, logMhMin=None, logMhMax=None):
        self.make_Msdists()  # first get stellar masses
        Ms_to_Mh = lambda logMs: SHMRs.DESI_1P({'model':'Psat'}).HSMR(logMs)()  # Use SHMR of Gao 2023 TODO: is the Psat model the best to use?

        # Bin halo masses into histogram of number distribution [unitless]
        self.N_logMh, self.logMh, self.logMhbins = self.make_N_q(Ms_to_Mh(np.log10(self.cat_ms)),
                                                            dq=dlogMh if dlogMh is not None else 0.01,  # arbitrary default Mh
                                                            qmin=logMhMin if logMhMin is not None else Ms_to_Mh(self.logMsMin), # default min Mh of (sub)sample from Studies
                                                            qmax=logMhMax if logMhMax is not None else Ms_to_Mh(self.logMsMax))  # default max Mh of (sub)sample from Studies
        self.dlogMh = self.logMh[1]-self.logMh[0]  # width of logMh bins
        self.dNdlogMh = self.N_logMh/self.dlogMh  # differential number distribution [Msol^-1]
        
        # TODO: add number density distribution [Msol^-1 Mpc^-3] (need volume info from Studies)

    # TODO: add 2D z/M distribution?



class Liu2025(BaseTargetData, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs
    path = f"{datapath}/Liu2025"  # path to data from zenodo.org/records/14706729
    subs = {
        'zbin' : ['all', 'z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        # 'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        # 'freq' : ['090', '150'],  # y map frequency (DR5)
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

        # Loading redshift distribution from file
        zcols = pd.read_csv(f"{datapath}/Zhou2023B/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get col names from Zhou2023
        self.zdf = pd.DataFrame(np.genfromtxt(f"{self.path}/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"), columns=zcols)  # Spectroscopic distributions of four sub-sample photometric redshift bins
        self.zs_df = (self.zdf.zmin+self.zdf.zmax).values/2  # define zs at center of bins

    # def get_beam(self):
    #     self.require(['ACTDR'])
    #     if self.ACTDR=='DR6': 
    #         ACTDR6 = Coulton2024()
    #         self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data
    #     elif self.ACTDR=='DR5': 
    #         self.require(['freq'])
    #         ACTDR5 = Naess2020({'freq':self.freq})
    #         self.beam_ells, self.beam_data = ACTDR5.beam_ells, ACTDR5.beam_data
    #         self.resp_ells, self.resp_data = ACTDR5.resp_ells, ACTDR5.resp_data

    def make_zdists(self, dz=None, zMin=None, zMax=None, **kwargs):
        self.require(['zbin'])
        zstr = f'bin_{self.zbin[-1]}' if self.zbin!='all' else 'all'
        self.n_df = self.zdf[f'{zstr}_combined'].values /u.deg**2
        self.dndz_df = self.n_df /(self.zs_df[1]-self.zs_df[0])/u.dex 
        zmin = zMin if zMin is not None else self.zs_df.min()
        zmax = zMax if zMax is not None else self.zs_df.max()
        self.dz = dz if dz is not None else self.zs_df[1]-self.zs_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)
        self.dndz = np.interp(self.z, self.zs_df, self.dndz_df)
        
        self.dNdz = self.dndz*(self.area)
        self.n_z = self.dndz*self.dz*u.dex
        self.N_z = self.dNdz*self.dz*u.dex

    def get_dndlogm(self):
        pass
    
    
    
    
    
    
    
    
    
class Schaan2021(BaseTargetData, Studies.Schaan2021):  # ACT DR5 maps stacked on SDSS BOSS DR10
    path = f"{datapath}/Schaan2021"  # path to data, shared by author
    subs = {
        'sample': ['cmass', 'lowz'],}
    
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
        self.bigdata = np.loadtxt('/global/homes/c/cpopik/Data/Schaan2021/catalog.txt')
        
    def get_zdist(self, dz, zMin, zMax):
        self.cat_zs = self.bigdata[:, 2]
        self.N_z, self.z, self.zbins = self.make_N_q(self.cat_zs, 
                                                    dq=dz if dz is not None else 0.01,  # arbitrary default dz
                                                    qmin=zMin if zMin is not None else self.zMin,  # default min z of (sub)sample from Studies
                                                    qmax=zMax if zMax is not None else self.zMax)  # default max z of (sub)sample from Studies
        self.dz = self.z[1]-self.z[0]  # width of z bins
        self.dNdz = self.N_z/self.dz  # differential number distribution [dex^-1]
        self.n_z = self.N_z/(self.area*u.deg**2)  # number density distribution [deg^-2 dex^-1]
        self.dndz = self.n_z/self.dz  # differential number density distribution [deg^-2 dex^-1]

    def get_Msdist(self, dlogMs, logMsMin, logMsMax):
        self.cat_Ms = self.bigdata[:, 18]
        self.N_logMs, self.logMs, self.logMsbins = self.make_N_q(np.log10(self.cat_Ms), 
                                                    dq=dlogMs if dlogMs is not None else 0.01,  # arbitrary default dlogMs
                                                    qmin=logMsMin if logMsMin is not None else self.logMsMin,  # default min logMs of (sub)sample from Studies
                                                    qmax=logMsMax if logMsMax is not None else self.logMsMax)  # default max logMs of (sub)sample from Studies
        
    def get_Mhdist(self, dlogMh, logMhMin, logMhMax):
        self.cat_Mh = self.bigdata[:, 20]
        self.N_logMh, self.logMh, self.logMhbins = self.make_N_q(np.log10(self.cat_Mh), 
                                                    dq=dlogMh if dlogMh is not None else 0.01,  # arbitrary default dlogMh
                                                    qmin=logMhMin if logMhMin is not None else self.logMhMin,  # default min logMh of (sub)sample from Studies
                                                    qmax=logMhMax if logMhMax is not None else self.logMhMax)  # default max logMh of (sub)sample from Studies


    
    
    
    
    
    
    
    
    
    
    
    
    
    


class Jenna_Catalog(BaseTargetData, Studies.Jenna_Catalog):
    path = "/global/homes/c/cpopik/Data/"  # location of data, provided by Jenna
    subs = {'masstype': ['Mstar', 'M200c', 'Mvir'], # Mass type (column names)
            }
    
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.dfdata = pd.read_csv(f"{self.path}/ACT_DR6_DESI_Y1Iron_LRGs_valid.csv")  # import datafarme
        
        self.get_dists()

    def get_dists(self, dlogM=0.01, dz=0.05, **kwargs): 
        self.require(['masstype'])
        logM, z = np.log10(self.dfdata[self.masstype]), self.dfdata.z
        dNdlogMdz, dNdlogM, self.dNdz, logM, self.z = self.dNdq1dq2_cat(logM, dlogM, z, dz)
        if self.masstype=='Mstar': self.logMs, self.dNdlogMs, self.dNdlogMs2D = logM, dNdlogM, dNdlogMdz*dz*u.dex  # if stellar mass, define the arrays as star
        else: self.logMh, self.dNdlogMh, self.dNdlogMh2D = logM, dNdlogM, dNdlogMdz*dz*u.dex  # if halo mass, define the arrays as halo
        self.dndz = self.dNdz/self.area
        
        
        
        
        


class Popik2025(BaseTargetData, Studies.Popik2025):  # TODO: In progress
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


        # Taking properties from Zhou
        Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
        for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
            setattr(self, val, getattr(Zhou, val))




class Hadzhiyska2025(BaseTargetData, Studies.Hadzhiyska2025):  # Stacked kSZ measurement of ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
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






class Gao2023(BaseTargetData, Studies.Gao2023):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    path = f"{datapath}/Gao2023"
    subs = {'sample':['LRG', 'ELG']}  # Galaxy Sample

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])

    def make_zMsdists(self, dz=None, zMin=None, zMax=None, dlogMs=None, logMsMin=None, logMsMax=None):
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z_df = (zbins[1:]+zbins[:-1])/2
        self.dz_df = self.z_df[1]-self.z_df[0]
        
        # Read the plot data from the files
        self.logMs_df = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.n_logMs_z_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]
        
        self.dndzdlogMs_df = self.n_logMs_z_h3 *self.h**3 /u.Mpc**3 /self.dz_df/u.dex**2 
        
        hmod = HaloModels.astropy_model(**Studies.Gao2023.info)
        Vcoms = (hmod.Vcom(self.z_df+self.dz_df/2)-hmod.Vcom(self.z_df-self.dz_df/2)) *(self.area/(4*np.pi*u.sr).to(u.deg**2))  # Calculate non-comoving shell for every z
        self.dNdzdlogMs_df = self.dndzdlogMs_df * Vcoms[:, None]
        
        dNinterp = RegularGridInterpolator((self.z_df, self.logMs_df), self.dNdzdlogMs_df,bounds_error=False, fill_value=0)

        zmin = zMin if zMin is not None else self.z_df.min()
        zmax = zMax if zMax is not None else self.z_df.max()
        self.dz = dz if dz is not None else self.z_df[1]-self.z_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)

        logMsMin = logMsMin if logMsMin is not None else self.logMs_df.min()
        logMsMax = logMsMax if logMsMax is not None else self.logMs_df.max()
        self.dlogMs = dlogMs if dlogMs is not None else self.logMs_df[1]-self.logMs_df[0]
        self.logMs = np.arange(logMsMin, logMsMax+self.dlogMs, self.dlogMs)

        zgrid, logMsgrid = np.meshgrid(self.z, self.logMs, indexing='ij')
        self.dNdzdlogMs = dNinterp(np.column_stack([zgrid.ravel(), logMsgrid.ravel()])).reshape(len(self.z), len(self.logMs)) / u.dex**2
        
        self.dNdogMs_z = self.dNdzdlogMs *self.dz*u.dex
        self.N_z = np.trapz(self.dNdogMs_z, self.logMs*u.dex)
        self.n_z = self.N_z / self.area
        self.dNdz = self.N_z / self.dz / u.dex
        self.dndz = self.dNdz / self.area
        
        self.N_z_logMs = self.dNdogMs_z *self.dlogMs*u.dex
        

    # def dndlogmstar(self, hh=0.71, **kwargs):  # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    #     return self.dndlogmstar_h3*hh**3
    
    # def dNdz(self, **cosmopars):
    #     return np.trapz(self.dndlogmstar(**cosmopars), self.logmstar)*self.volumes(**cosmopars)/(self.z[1]-self.z[0])


class Kou2023(BaseTargetData, Studies.Kou2023):
    path = "/global/homes/c/cpopik/Data/Kou2023"  # path to data, taken from plots using webplotdigitizer
    subs = {'mbin':['M1', "M2", "M3", "M4"]}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['mbin'])




class Zhou2023(BaseTargetData, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B"  # path to data from zenodo.org/records/8319955
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }

    def __init__(self, inputsdict, **inputvars):       
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'sample', 'hemisphere'])

        self.get_dndz()

    def get_dndz(self):
        samp = {'main':'main', 'ext':'extended'}[self.sample]
        cols = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get columns from first row
        zdf = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", skiprows=1, names=cols)  # format into dataframe
        self.z = (zdf.zmin.values+zdf.zmax.values)/2
        pzstr = f'bin_{self.zbin[-1]}_{self.hemisphere}' if self.zbin!='all' else 'all'  # get name of column corresponding on bin
        self.dz = self.z[1]-self.z[0]
        self.dndz = zdf[pzstr].values//self.dz /u.deg**2/u.dex
        self.dNdz = self.dndz * (self.area*u.deg**2)




class Amodeo2021(BaseTargetData, Studies.Amodeo2021):  # Inference on BOSS DR10 stacked ACT DR5 (arxiv.org/abs/2009.05558)
    path = '/global/homes/c/cpopik/Data/Amodeo2021/'  # path to data, taken from plots using webplotdigitizer and various repos
    subs = {'prof': ['Amodeo', 'Battaglia', 'TNG'],
                'units': ['cosmo', 'cgs']}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['model', 'units'])

    def get_dNdlogmhalo(self, dlogmhalo=0.05, dlogmstar=0.05):
        massdist = np.loadtxt(f'{self.path}/mass_distrib.txt')        
        self.dNdlogMh, self.logMh, Mhalobins = self.dNdq_cat(np.log10(massdist), dlogmhalo)  # Get 1D hist for z

        massdiststar = SHMRs.Kravstov2018({'mdef':'200c', 'scatter':'S'}).HSMR(np.log10(massdist))()  # steal default SHMR
        self.dNdlogMs, self.logMs, Msbins = self.dNdq_cat(massdiststar, dlogmstar)  # Get 1D hist for z





class SDSSBOSS(BaseTargetData, Studies.Ahn2013Alam2015):  # (Ahn+ 2013, arxiv.org/abs/1307.7735, Alam+ 2015, https://arxiv.org/abs/1501.00963)
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
        
    def get_catalog(self):  # import catalog
        self.require(['group'])  # base info
        vers = {'DR10':['v1_0', '5_12'], 'DR12':['v1_1', '7_0']}[self.DR]
        self.path = f"{self.path}/dr{self.DR[2:]}/boss/spectro/redux/galaxy/{vers[0]}"  # locate folder
            
        if self.group=='portsmouth': 
            self.require(['template', 'IMF'])
            fname, mcolname = f"{self.group}_stellarmass_{self.template}_{self.IMF}-v5_{vers[1]}.fits.gz", 'LOGMASS'
        elif self.group=='wisconsin': 
            self.require(['pop'])
            fname, mcolname = f"{self.group}_pca_{self.pop}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'
        elif self.group=='granada': 
            self.require(['template', 'time', 'dust'])
            fname, mcolname = f"{self.group}_fsps_{self.IMF}_{self.time}_{self.dust}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'

        # Fetch the data and rename the mass column, sdss4.org/dr17/spectro/galaxy_portsmouth/
        self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

        # Select the correct galaxy sample using the bitmasks in sdss3.org/dr10/algorithms/bitmask_boss_target1.php, sdss3.org/dr10/algorithms/bitmasks.php, sdss4.org/dr17/algorithms/bitmasks/, skyserver.sdss.org/dr19/MoreTools/browser/
        decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
        self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
        self.bitmasks = {'CMASS':7, 'LOWZ':0}

    def make_dNdz(self, dz=0.1, zMin=None, zMax=None, **kwargs):
        self.require(['galaxy'])
        dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
        self.N_z, self.z, zbins = self.make_N_q(dfdata.Z, dz, zMin, zMax)
        self.dz = dz
        self.dNdz = self.N_z / self.dz/u.dex
        self.dndz = self.dNdz/ self.area/u.deg**2
    
    def get_dNdlogMs(self, dlogMs=0.05, logMsmin=None, logMsmax=None):
        self.require(['galaxy'])
        dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
        dNdlogMs, logMs, logMsbins = self.make_N_q(dfdata.LOGM, dlogMs, logMsmin, logMsmax)
        return dNdlogMs, logMs, dlogMs
        
    def get_dNdlogMs(self, dlogMs=0.05, dz=0.05, zmin=None, zmax=None, logMsmin=None, logMsmax=None):
        self.require(['galaxy'])
        dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
        dNdlogMsdz, dNdlogMs, dNdz, logMs, z = self.make_N_q1_q2(dfdata.LOGM, dlogMs, self.dfdata.Z, dz)
        dNdlogMs2D = dNdlogMsdz*dz*u.dex
        return dNdlogMs2D, z, logMs, dz, dlogMs





