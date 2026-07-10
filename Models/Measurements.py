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


datapath = "/global/homes/c/cpopik/Data"  # path to data


class BaseMeasurement:
 def __init__(self):
     pass


# class Sailer2025:  # TODO: In progress DESI LS LRGs correlated with Planck PR4&ACT DR6 Lensing (Sailer+ 2025, arxiv.org/abs/2407.04607)
#     path = f"{datapath}/Sailer2025"  # Path to data from zenodo.org/records/12613408

# class Qu2025:  # TODO: In progress DESI LS Galaxies correlated with Planck PR4&ACT DR6 Lensing (Qu+ 2025, arxiv.org/abs/2410.10808)
#     path = f"{datapath}/Qu2025"  # Path to data from zenodo.org/records/13844390
    
# class Maus2025:  # TODO: In progress DESI DR1 Galaxies correlated with Planck PR4&ACT DR6 Lensing (Maus+ 2025, arxiv.org/abs/2505.20656)
#     path = f"{datapath}/Maus2025"  # Path to data from zenodo.org/records/17636841

class Popik2026(BaseMeasurement, Studies.Popik2026):  # TODO: In progress
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


        self.get_meas()

        # Taking properties from Zhou
        Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
        for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
            setattr(self, val, getattr(Zhou, val))

    def get_meas(self):
        with h5py.File(f"{self.path}/ACTDR6DESILRG_Spectra_testnew.h5", 'r') as f:
            self.ell = f['ell'][()]
            self.Cgg_data = f[f'gxg/{self.zbin[-1]}'][()]
            self.Cgy_data = f[f'gxy/{self.zbin[-1]}/{self.deproj}/{self.TCIB}/{self.beta}'][()]
            self.Cyy_data = f[f'yxy/{self.deproj}/{self.TCIB}/{self.beta}'][()]

            self.Cgy_err = np.abs(self.Cgy_data)/10000*self.ell  # TODO: placeholder




class Moore2026(BaseMeasurement, Studies.Moore2026):
    path = f"{datapath}/Moore2026"
    # subs = {'sample': ['BGS', 'LRG_deprojected', 'optical', 'LRG_raw'], # Boryana BGS sample, Yun-Hsuin's optical sample, LRG
    #         'deproj': ['_f090', '_f150', '_ILC', '_ILC_CIB_deproj', '_ILC_dB_deproj'], # deprojection method
    #         'richbin': ['all', '10', '20'], # richness bin, only for optical
    #         'mbin': ['10.00', '10.25', '10.50', '10.75', '11.00', '11.25'], # stellar mass bin for BGS sample
    #         'Lbin': ['L36', 'L48', 'L60', 'L79', 'L98', 'L36D', 'L48D', 'L60D', 'L79D'], # luminosity bin, just for LRGs, cumulative and disjoint
    #         }
    
    subs = {'sample': ['bgs', 'opt', 'lrg'],
            'prof': ['fiducial_ilc','ilc_CIB_deproj', 'ilc_cib_deproj', 'ilc_dB_deproj', 'f090_raw', 'f150_raw', 'f090_corrected', 'f150_corrected', 'f220_raw', 'f090', 'f150'],
            'bin': ['L35', 'L47', 'L59', 'L78', 'L98', 'L35D', 'L47D', 'L59D', 'L78D', '10.0', '10.25', '10.5', '10.75', '11.0', '11.25', 'all', '10', '20', 'L36', 'L48', 'L60', 'L79', 'L98', 'L36D', 'L48D', 'L60D', 'L79D'], # luminosity bin, just for LRGs, cumulative and disjoint
            'radio': ['1', '2p7', '2p1','incl', '2p4']
            }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
        self.get_data()
        
    def get_data(self):
        self.require(['sample', 'prof', 'bin'])
        
        if self.radio!='incl':
            if self.prof in ['fiducial_ilc', 'ilc_cib_deproj', 'ilc_dB_deproj', 'f090', 'f150']:
                datadf = pd.read_csv(f"{self.path}/for_chad_20260519/{self.sample}_{self.prof}_profiles_yarcmin2_radio_clean_{self.radio}arcmin.csv")
            else:
                datadf = pd.read_csv(f"{self.path}/for_chad_20260519/{self.sample}_{self.prof}_profiles_radio_clean_{self.radio}arcmin.csv")
        else:
            if self.prof in ['fiducial_ilc', 'ilc_CIB_deproj', 'ilc_dB_deproj', 'f090', 'f150']:
                datadf = pd.read_csv(f"{self.path}/profiles/{self.sample}_{self.prof}_profiles_yarcmin2.csv")
            else:
                datadf = pd.read_csv(f"{self.path}/profiles/{self.sample}_{self.prof}_profiles.csv")

        rvals = np.unique([col.split('_')[1] for col in datadf.columns[1:]])
        self.R = np.array([np.float64(rval.replace("p", ".")) for rval in rvals]) *u.arcmin
        if self.prof in ['fiducial_ilc', 'ilc_CIB_deproj', 'ilc_dB_deproj', 'f090', 'f150', 'ilc_cib_deproj']: areafac = u.arcmin**2
        else: areafac = np.pi*self.R**2
        
        data, err={},{}
        for i,dfbin in enumerate(datadf.iloc[:,0]):
            data[f'{dfbin}']= np.array([datadf[f'dT_{rval}_arcmin'][i] for rval in rvals])*areafac
            err[f'{dfbin}']= np.array([datadf[f'dT_{rval}_arcmin_err'][i] for rval in rvals])*areafac
            
        if self.prof in ['fiducial_ilc', 'ilc_CIB_deproj', 'ilc_dB_deproj', 'f090', 'f150', 'ilc_cib_deproj']: self.tSZ_data, self.tSZ_err = data[self.bin], err[self.bin]
        else: self.tSZ_data, self.tSZ_err = data[self.bin] *u.uK, err[self.bin] *u.uK
        
        self.tSZ_cov = np.diag(self.tSZ_err**2)
        
        if '090' in self.prof: self.freq = 90*u.GHz
        elif '150' in self.prof: self.freq = 150*u.GHz
        elif '220' in self.prof: self.freq = 220*u.GHz
        else: pass

        
            
        

        
    # def datatest(self):
    #     self.require(['sample', 'deproj'])
    #     file = pd.read_csv(f'{self.path}/{self.sample}_AP_results.csv')
    #     self.file=file

    #     # these are all aperture AVERAGED measurements, so to get them in arcmin^2 units multiple by piR^2
        
    #     if self.sample=='LRG_raw':
    #         self.require(['Lbin'])
    #         if self.deproj=='':
    #             self.require(['freq'])
    #             self.tSZ_data = file[file.iloc[:, 0]==self.Lbin]['dy'+self.freq].values
    #             self.tSZ_err = file[file.iloc[:, 0]==self.Lbin]['dyerr'+self.freq].values
    #         else:
    #             self.tSZ_data = file[file.iloc[:, 0]==self.Lbin]['dy'+self.deproj].values *u.uK
    #             self.tSZ_err = file[file.iloc[:, 0]==self.Lbin]['dyerr'+self.deproj].values *u.uK
    #         self.R = np.array([2.1])*u.arcmin
        
    #     elif self.sample=='LRG_deprojected':
    #         self.require(['Lbin'])
    #         if self.deproj=='':
    #             self.require(['freq'])
    #             self.tSZ_data = file[file.iloc[:, 0]==self.Lbin]['dy'+self.freq].values
    #             self.tSZ_err = file[file.iloc[:, 0]==self.Lbin]['dyerr'+self.freq].values
    #         else:
    #             self.tSZ_data = file[file.iloc[:, 0]==self.Lbin]['dy'+self.deproj].values
    #             self.tSZ_err = file[file.iloc[:, 0]==self.Lbin]['dyerr'+self.deproj].values
    #         self.R = np.array([2.1])*u.arcmin
    #     elif self.sample=='BGS':
    #         self.require(['mbin'])
    #         self.tSZ_data = file[file.iloc[:, 0]==np.float64(self.mbin)]['dy'+self.deproj].values
    #         self.tSZ_err = file[file.iloc[:, 0]==np.float64(self.mbin)]['dyerr'+self.deproj].values
    #         self.R = np.array([2.7]) *u.arcmin
    #     elif self.sample=='optical':
    #         self.require(['richbin'])
    #         self.tSZ_data = file[file.iloc[:, 0]==self.richbin]['dy'+self.deproj].values
    #         self.tSZ_err = file[file.iloc[:, 0]==self.richbin]['dyerr'+self.deproj].values
    #         self.R = np.array([2.1]) *u.arcmin
            
    #     self.tSZ_cov = np.array([self.tSZ_err]) *self.tSZ_err.unit



class RiedGuachalla2025(BaseMeasurement, Studies.RiedGuachalla2025):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs (arxiv.org/abs/2503.19870)
    path = f"{datapath}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass bin
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['bin'])

        self.get_meas()  # get the measurements

    def get_meas(self):
        # TODO: Add more of the plots as options
        
        kSZ_fid = dict(np.load(f"{self.path}/fig8_fiducial.npz")) # The measured stacked kSZ in μK arcmin2 for varying CAP filters with radius R
        kSZ_zbins = dict(np.load(f"{self.path}/fig11_ksz_z.npz"))  # Mean stacked kSZ profiles for the different redshift bins
        kSZ_mbins = dict(np.load(f"{self.path}/fig12_ksz_mass.npz"))  # kSZ stacked profiles for the different stellar mass bins, denoted by mass

        if self.bin=='all':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_fid['R'] *u.arcmin, kSZ_fid['DESIxACT'] *u.uK*u.arcmin**2, kSZ_fid['errors_DESIxACT'] *u.uK*u.arcmin**2

        elif self.bin[0]=='z':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_zbins['R'] *u.arcmin, kSZ_zbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_zbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        elif self.bin[0]=='m':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_mbins['R'] *u.arcmin, kSZ_mbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_mbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        self.cormat = dict(np.load(f"{self.path}/fig18_cor.npz"))['cor']
        self.TkSZ_cov = np.diag(self.TkSZ_err) @ self.cormat @ np.diag(self.TkSZ_err)
  



class Hadzhiyska2025(BaseMeasurement, Studies.Hadzhiyska2025):  # Stacked kSZ measurement of ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
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

        simsload = np.load(f"{self.path}/Fig2_sim.npz")  # Stacked kSZ signal of Main sample, all z bins, and from TNG and Illustris models
        self.R = simsload['theta_arcmins'] *u.arcmin
        self.kSZ_Illustris1 = simsload['gas_illustris'] *u.uK*u.arcmin**2
        self.kSZ_TNG300 = simsload['dm_tng']  *u.uK*u.arcmin**2
        self.kSZ = simsload['signal']  *u.uK*u.arcmin**2

        samplestr = {'main': '', 'extended': 'extended_', 'all': ''}[self.sample]
        corrstr = {'corrected':'corr', 'uncorrected':''}[self.corr]
        zstr = {'nocut': '', 'cut': 'sigmaz0.05000_'}[self.zoutcut]
        filename = f"{self.path}/Fig1_Fig8_{samplestr}dr10_allfoot_perbin_{zstr}dr6_{corrstr}pzbin{self.zbin[-1]}.npz"

        self.kSZ_data = np.load(filename)['prof'] *u.uK*u.arcmin**2
        self.kSZ_cov = np.load(filename)['cov'] *(u.uK*u.arcmin**2)**2
        self.kSZ_err = np.diag(self.kSZ_cov)**0.5


class Liu2025(BaseMeasurement, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs (Liu+ 2025, arxiv.org/abs/2502.08850)
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
        elif self.ACTDR=='DR5': 
            self.require(['zbin', 'aper', 'freq'])
            
        try: self.get_meas_shared()
        except: pass

        self.get_meas()

    def get_meas(self):
        if self.ACTDR=='DR6':  # ACT DR6 stacked y profiles, fiducial and using deprojection method/values:
            if self.dp=='beta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig3.csv")  # using CIB deprojected y maps
            elif self.dp=='dBeta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig4.csv")  # using CIB & dBeta moment deprojected y maps
            elif self.dp=='beta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig10.csv")  # using CIB deprojected y maps, T=24.0K
            elif self.dp=='dBeta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig11.csv")  # using CIB & dBeta moment deprojected y maps, T=24.0K
            
            dBetastr = f"Beta_{self.Beta}" if self.Beta!='fiducial' else 'fiducial'
            
            self.R = y['RApArcmin'].values[:-1] *u.arcmin
            self.tSZ_data= y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}"].values[:-1] *u.arcmin**2
            self.tSZ_err = y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}_err"].values[:-1] *u.arcmin**2

        elif self.ACTDR=='DR5':  # ACT DR5 stacked y profiles
            if self.aper=='CAP': y = pd.read_csv(f"{self.path}/fig12.csv")  # using standard CAP
            elif self.aper=='RingRing': y = pd.read_csv(f"{self.path}/fig13.csv")  # using ring-ring filter
            
            self.R = y['RApArcmin'].values *u.arcmin
            self.tSZ_data = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}"].values *u.arcmin**2
            self.tSZ_err = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}_err"].values *u.arcmin**2
            
    def get_meas_shared(self):
        if self.ACTDR=='DR6' and self.TCIB=='10.7':
            dBetastr = f"dBeta_{self.Beta}_{self.TCIB}" if self.Beta!='fiducial' else 'fiducial'
            y = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/diskring_tsz_uniformweight_measured.txt").T
            ycov = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/cov_diskring_tsz_uniformweight_bootstrap.txt").T

            self.R = y[0] *u.arcmin
            self.tSZ_data = (y[1] *u.sr).to(u.arcmin**2)
            self.tSZ_err = (y[2] *u.sr).to(u.arcmin**2)
            self.tSZ_cov = (ycov *u.sr**2).to(u.arcmin**4)

        

    
    
class Kou2023(BaseMeasurement, Studies.Kou2023):
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


class Zhou2023(BaseMeasurement, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
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
        
    def get_meas(self):
        samp = {'main':'fid', 'ext':'ext'}[self.sample]

        Cls = json.load(open(f"{self.path}/combined_cls.json", "r"))
        self.ell = np.array(Cls['ell'])
        self.Cgg_data = np.array(Cls[f'cls_{samp}'][f's0{self.zbin[-1]}'])
        
        self.Wggs = json.load(open(f"{self.path}/combined_wth.json", "r"))
        self.thetas = np.array(self.Wggs['theta(deg)'])  # [deg]
        self.Wgg_data = np.array(self.Wggs[f'wth_{samp}'][f's0{self.zbin[-1]}'])  # angular correlation function




class Kusiak2022(BaseMeasurement, Studies.Kusiak2022):  # HOD for unWISE galaxies and Planck lensing arxiv.org/abs/2203.12583
    path = "/global/homes/c/cpopik/Data/Kusiak2022"  # path to data, taken from plots using webplotdigitizer
    subs = {'sample':['Blue', 'Green', 'Red']}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])
        
        self.get_meas()
        _, _ = self.get_dNdz(20)
        
        

        



class White2022(BaseMeasurement, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing (White+ 2022, arxiv.org/abs/2111.09898)
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

        self.require(['zbin'])
            
        self.get_meas()

    def get_meas(self):
        self.ell, self.Cgg_data = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cls.txt").T[0:2]  # ells and measured angular auto-spectra
        self.ell_model, self.Cgg_data_model = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_mod.txt").T[0:2]  # smooth model used to calculate cov mat
        
        self.C_covcomb = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cov.txt")  # combined covariance matrix of auto and cross
        self.Cgg_cov = self.C_covcomb[:int(self.ell.size)]  # TODO: lazely assuming i can do this
        self.Cgg_err = np.diag(self.Cgg_cov)**0.5
        
        # weights of all multipoles used to get effective multipole ell, ell_effective = np.sum(weights*ells)
        self.auto_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wla.txt")  # window function matrix for auto-spec
        self.cross_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wlx.txt")  # window function matrix for cross-spec
        


class Amodeo2021(BaseMeasurement, Studies.Amodeo2021):  # Inference on BOSS DR10 stacked ACT DR5 (arxiv.org/abs/2009.05558)
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


        

class Schaan2021(BaseMeasurement, Studies.Schaan2021):  # ACT DR5 maps stacked on SDSS BOSS DR10 (Schaan+ 2021, arxiv.org/abs/2009.05557)
    path = f"{datapath}/Schaan2021"  # path to data, shared by author
    subs = {
        'sample': ['cmass', 'lowz'],
        'freq' : ['150', '090'],  # frequency band of obsevation [GHz]
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

        self.get_meas()  # get measurement

    def get_meas(self):
        self.require(['sample', 'freq'])

        measpath = f"{self.path}/{self.sample}_data_sharing_schaan21/f{self.freq}"  # each meas in different folder
        if self.sample=='cmass':
            self.R = np.genfromtxt(f"{measpath}/diskring_tsz_varweight_measured.txt").T[0] *u.arcmin
            self.kSZ_data, self.kSZ_err = (np.genfromtxt(f"{measpath}/diskring_ksz_varweight_measured.txt").T[1:] *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.tSZ_data, self.tSZ_err = (np.genfromtxt(f"{measpath}/diskring_tsz_varweight_measured.txt").T[1:] *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.kSZ_cov = (np.genfromtxt(f"{measpath}/cov_diskring_ksz_varweight_bootstrap.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)
            self.tSZ_cov = (np.genfromtxt(f"{measpath}/cov_diskring_tsz_varweight_bootstrap.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)

        elif self.sample=='lowz':
            freqstr = str(int(self.freq))
            self.R = np.genfromtxt(f"{measpath}/ksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T[0] *u.arcmin
            self.kSZ_data = (np.genfromtxt(f"{measpath}/ksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt") *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.tSZ_data = (np.genfromtxt(f"{measpath}/tsz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt") *u.uK*u.sr).to(u.uK*u.arcmin**2)
            self.kSZ_cov = (np.genfromtxt(f"{measpath}/covksz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)
            self.tSZ_cov = (np.genfromtxt(f"{measpath}/covtsz_lowz_kendrick_pactf{freqstr}daynight20200228maskgal60r2.txt").T *(u.uK*u.sr)**2).to((u.uK*u.arcmin**2)**2)

        for val in ['kSZ', 'tSZ']:  # get errors from covariance matrices
            setattr(self, f'{val}_err', np.diag(getattr(self, f'{val}_cov'))**0.5) 

        # Convert to y units

        # convfac = 1/HaloModels.y_to_uK(np.float32(self.freq)*u.GHz, self.T_CMB)
        # self.y_data = self.TtSZ_data*convfac
        # self.y_cov = self.TtSZ_cov*convfac**2








class Naess2020(BaseMeasurement, Studies.Naess2020):  # ACT DR5 (Naess 2020, arxiv.org/abs/2007.07290)
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



class Koukoufilippas2020(BaseMeasurement, Studies.Koukoufilippas2020):  # arxiv.org/abs/1909.09102
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
            
            
class CAMELShalo(BaseMeasurement):  # Random halo
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