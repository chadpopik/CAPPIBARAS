"""
Collection of measurements/covariances from different stacking/cross-correlation studies and corresponding components needed for forward modelling and inference, such as the beam and response. 

- TODO 1: Find beam and response for ACT DR5
- TODO 2: Add info about the measurement like window function/aperture photometry info or m/z selection
"""

import os
import numpy as np
import astropy.units as u
import pandas as pd

datapath = "/global/homes/c/cpopik/CAPPIBARAS/Data/"

class BaseData:
    def checkspefs(self, spefs, required):  # Check if the spef is in the list
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
                
    def get_beam_ACTDR5(self):
        self.beamfile = f"{datapath}/ACTDR5/beams/act_planck_dr5.01_s08s18_f{self.freq}_daynight_beam.txt"
        self.respfile = f"{datapath}/ACTDR5/responses/act_planck_dr5.01_s08s18_AA_f{self.freq}_daynight_response_tsz.txt"
    
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = -self.resp_data  # Is negative for some reason
        
    def get_beam_ACTDR6(self):  # TODO 1: These are old DR5 beams/responses, need ones for DR6
        self.beamfile = f"{datapath}/ACTDR5/beams/act_planck_dr5.01_s08s18_f150_daynight_beam.txt"
        self.respfile = f"{datapath}/ACTDR5/responses/act_planck_dr5.01_s08s18_AA_f150_daynight_response_tsz.txt"
    
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]
        self.resp_data = -self.resp_data  # Is negative for some reason    



class RiedGuachalla2025(BaseData):  # ACT DR6 maps stacked on DESI Y1 LRGs (arxiv.org/abs/2503.19870)
    zbins = ['all', '1', '2', '3', '4']
    mbins = ['all', '1', '2', '3', '4']

    def __init__(self, spefs):    
        self.checkspefs(spefs, required=['zbin', 'mbin'])
        self.get_meas()
        self.get_beam_ACTDR6()
        
    def get_meas(self):
        kszstacked = dict(np.load(f"{datapath}/RiedGuachalla2025/fig8_fiducial.npz"))
        self.thetas = kszstacked['R']  # [arcmin]
        self.kSZdata, self.kSZerr = kszstacked['DESIxACT'], kszstacked['errors_DESIxACT']

        if self.zbin!='all':
            if self.mbin!='all': raise NameError("Not available")
            kszstacked = dict(np.load(f"{datapath}/RiedGuachalla2025/fig11_ksz_z.npz"))
            self.kSZdata, self.kSZerr = kszstacked[f'z_{self.zbin}'], kszstacked[f'z_{self.zbin}_error']
        
        elif self.mbin!='all':
            kszstacked = dict(np.load(f"{datapath}/RiedGuachalla2025/fig12_ksz_mass.npz"))
            self.kSZdata, self.kSZerr = kszstacked[f'mass_{self.mbin}'], kszstacked[f'mass_{self.mbin}_error']


class Hadzhiyska2025(BaseData):  # ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
    zbins = ['1', '2', '3', '4']
    samples = ['main', 'extended', 'all']
    zoutcuts = ['nocut', 'cut']
    corrs = ['corrected', 'uncorrected']
    
    def __init__(self, spefs):       
        self.checkspefs(spefs, required=['zbin', 'sample', 'corr', 'zoutcut'])
        self.get_meas()
        self.get_beam_ACTDR6()
        
    def get_meas(self):
        self.thetas = np.load(f"{datapath}/Hadzhiyska2024/Fig2_sim.npz")['theta_arcmins']
        self.Tksz_Illustris1 = np.load(f"{datapath}/Hadzhiyska2024/Fig2_sim.npz")['gas_illustris']
        self.Tksz_TNG300 = np.load(f"{datapath}/Hadzhiyska2024/Fig2_sim.npz")['dm_tng']
        self.Tksz = np.load(f"{datapath}/Hadzhiyska2024/Fig2_sim.npz")['signal']
        
        samplestr = {'main': '', 'extended': 'extended_', 'all': ''}[self.sample]
        corrstr = {'corrected':'corr', 'uncorrected':''}[self.corr]
        zstr = {'nocut': '', 'cut': 'sigmaz0.05000_'}[self.zoutcut]
        
        filename = f"{datapath}/Hadzhiyska2024/Fig1_Fig8_{samplestr}dr10_allfoot_perbin_{zstr}dr6_{corrstr}pzbin{self.zbin}.npz"
        self.kSZdata = np.load(filename)['prof']
        self.kSZcov = np.load(filename)['cov']
        self.kSZerr = np.diag(self.kSZcov)**0.5


class Liu2025(BaseData): # ACT DR6 maps stacked on DESI LRGs for cross-correlation (arxiv.org/abs/2502.08850)
    zbins = ['1', '2', '3', '4']
    Betas = ['fiducial', '1.2', '1.4', '1.6']
    TCIBs = ['10.7', '24.0']
    dps = ['beta', 'dBeta']
    DRs = ['DR5', 'DR6']
    apers = ['CAP', 'RingRing']
    freqs = ['90', '150']

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['zbin', 'DR'])
        if self.DR=='DR6': 
            self.checkspefs(spefs, required=['Beta', 'dp', 'TCIB'])
            self.get_meas()
        elif self.DR=='DR5': 
            self.checkspefs(spefs, required=['aper', 'freq'])
            self.get_meas_DR5()
            
        if os.path.isdir(f"{datapath}/Liu2025/shared"):
            self.get_meas_shared()
            
        self.get_beam_ACTDR6()
        
    def get_meas(self):
        if self.dp=='beta' and self.TCIB=='10.7': fname = "fig3.csv"
        elif self.dp=='dBeta' and self.TCIB=='10.7': fname = "fig4.csv"
        elif self.dp=='beta' and self.TCIB=='24.0': fname = "fig10.csv"
        elif self.dp=='dBeta' and self.TCIB=='24.0': fname = "fig11.csv"
        
        dfmeas = pd.read_csv(f"{datapath}/Liu2025/{fname}")
        dBetastr = f"Beta_{self.Beta}" if self.Beta!='fiducial' else 'fiducial'
        
        self.thetas = dfmeas['RApArcmin'].values[:-1]  # [arcmin]
        self.tSZdata, self.tSZerr = dfmeas[f"pz{self.zbin}_act_dr6_{dBetastr}"].values[:-1], dfmeas[f"pz{self.zbin}_act_dr6_{dBetastr}_err"].values[:-1]  # [arcmin^2]
        
    def get_meas_DR5(self):
        if self.aper=='CAP': fname = "fig12.csv"
        elif self.aper=='RingRing': fname = "fig13.csv"
        dfmeas = pd.read_csv(f"{datapath}/Liu2025/{fname}")
        
        self.thetas = dfmeas['RApArcmin']  # [arcmin]
        self.tSZdata, self.tSZerr = dfmeas[f"pz{self.zbin}_act_dr5_f{int(self.freq)}"], dfmeas[f"pz{self.zbin}_act_dr5_f{int(self.freq)}_err"]  # [arcmin^2]

    def get_meas_shared(self):
        # Find data and covariance files for bin and CIB method
        dBetastr = f"dBeta_{self.dBeta}_10.7" if self.dBeta!='fiducial' else 'fiducial'
        self.tSZdatafile = f"{datapath}/Liu2025_shared/DESI_pz{self.zbin}_act_dr6_{dBetastr}/diskring_tsz_uniformweight_measured.txt"
        self.tSZcovfile = f"{datapath}/Liu2025_shared/DESI_pz{self.zbin}_act_dr6_{dBetastr}/cov_diskring_tsz_uniformweight_bootstrap.txt"

        # Load in data and convert to arcmin^2
        self.thetas = np.genfromtxt(self.tSZdatafile).T[0]  # [arcmin]
        self.tSZdata_shared, self.tSZerr_shared = np.genfromtxt(self.tSZdatafile).T[1:3]*u.sr.to(u.arcmin**2)  # [ster]->[arcmin^2]
        self.tSZcov_shared = np.genfromtxt(self.tSZcovfile).T*u.sr.to(u.arcmin**2)**2  # [ster]->[arcmin^2]


class Schaan2021(BaseData):  # ACT DR5 maps stacked on SDSS BOSS DR10/DR12 (Schaan+ 2021, arxiv.org/abs/2009.05557)
    # General info about the measurement
    info = {'mhalomax': 1e14}

    freqs = ['150', '090']
    samples = ['cmass', 'lowz']

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['freq', 'sample'])
        self.get_meas()
        self.get_beam_ACTDR5()

    def get_meas(self):
        self.thetas = np.genfromtxt(f"{datapath}/Schaan2021_shared/cmass_data_sharing_schaan21/f150/diskring_tsz_varweight_measured.txt").T[0]  # [arcmin]
        path = f"{datapath}/Schaan2021_shared/{self.sample}_data_sharing_schaan21/f{self.freq}"

        if self.sample=='cmass':
            self.kSZdatafile, self.tSZdatafile = [f"{path}/diskring_{meas}_varweight_measured.txt" for meas in ['ksz', 'tsz']]
            self.kSZdata, self.kSZerr = np.genfromtxt(self.kSZdatafile).T[1:] *u.sr.to(u.arcmin**2)  # [muK*ster]->[muK*arcmin^2]
            self.tSZdata, self.tSZerr = np.genfromtxt(self.tSZdatafile).T[1:] *u.sr.to(u.arcmin**2) # [muK*ster]->[muK*arcmin^2]

            self.kSZcovfile, self.tSZcovfile = [f"{path}/cov_diskring_tsz_varweight_bootstrap.txt" for meas in ['ksz', 'tsz']]
            self.tSZcov = np.genfromtxt(self.kSZcovfile).T *u.sr.to(u.arcmin**2)**2  # [muK*ster]^2->[muK*arcmin^2]^2
            self.kSZcov = np.genfromtxt(self.tSZcovfile).T *u.sr.to(u.arcmin**2)**2  # [muK*ster]^2->[muK*arcmin^2]^2
    
        elif self.sample=='lowz':
            self.kSZdatafile, self.tSZdatafile = [f"{path}/{meas}_lowz_kendrick_pactf{str(int(self.freq))}daynight20200228maskgal60r2.txt" for meas in ['ksz', 'tsz']]
            self.kSZdata = np.genfromtxt(self.kSZdatafile)  # [muK*arcmin^2]
            self.tSZdata = np.genfromtxt(self.tSZdatafile) # [muK*arcmin^2]
            
            self.kSZcovfile, self.tSZcovfile = [f"{path}/cov{meas}_lowz_kendrick_pactf{str(int(self.freq))}daynight20200228maskgal60r2.txt" for meas in ['ksz', 'tsz']]
            self.tSZcov = np.genfromtxt(self.kSZcovfile).T # [muK*arcmin^2]^2
            self.kSZcov = np.genfromtxt(self.tSZcovfile).T # [muK*arcmin^2]^2
            
