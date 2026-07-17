


import os
import pandas as pd
from Models.Papers.PlotsTables import ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))


class Studies(BaseStudy):
    subs={}
    info = {'area': 16700*u.deg**2,  # assuming the same as XCorr LRGs
            }


class TargetDataInfoTable(ParamTable):  # per sample/bin best-fit values (LMin/LMax/MhMin/MhMax/MsMean/LMean/zMean/N/Nf150/logMsMin/logMsMean/lambdaMin/lambdaMean/logMhMean)
    def __init__(self, filename=f"{thispath}/target_data_info.csv"):
        super().__init__(filename)


class TargetData(BaseTargetData, Studies.Moore2026):
    path = f"{datapath}/Moore2026"  # location of data, provided by Jenna
    subs = {'masstype': ['M200c', 'Mvir'], # Mass type (column names)
            'sample': ['LRG', 'BGS', 'optical'],
            'bin': ['L36', 'L48', 'L60', 'L79', 'L98', 'L36D', 'L48D', 'L60D', 'L79D',
                    'M10', 'M1025', 'M105', 'M1075', 'M11', 'M1125',
                    'Full', 'L10', 'L20'],
            'radio': ['1.0', '2.7', '2.1','incl', '2.4']
            }
    info = {
    'area': 16700*u.deg**2,  # assuming the same as XCorr LRGs
    }

    UNITS = {'LMin': 1e10*u.Lsun, 'LMax': 1e10*u.Lsun, 'MhMin': 1e13*u.Msun,  # virial mass, 1e13 Msun
             'MhMax': 1e13*u.Msun, 'MsMean': 1e11*u.Msun, 'LMean': 1e10*u.Lsun}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        row = TargetDataInfoTable().getparams(sample=self.sample, bin=self.bin).to_dict()
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v): v = None
            if v is not None and k in self.UNITS: v = v*self.UNITS[k]
            setattr(self, k, v)

    def get_cat(self):
        self.require(['sample'])
        if self.radio=='incl':
            sampfiles = {
                'LRG': f"{self.path}/ACT_DR6_DESI_Y1Iron_LRGs_valid", # LRG sample from Yu-Lin's paper
                'BGS': f"{Hadzhiyska2026.path}/BGS_BRIGHT_full_noveto_vac_marvin_BGS_ACT_DR6_fixedTh2.10_delta_Ts", # BGS sample for Boryana's paper
                'optical': f"{self.path}/iskay_RM_ls10_iron_v1_CGz_cat_310525",  # Optical sample from Yun-Hsuin
                        }
            self.dfdata = pd.read_csv(f"{sampfiles[self.sample]}.csv")  # import datafarme
        else:
            sampfiles = {
                'LRG': f"{self.path}/for_chad_20260519/LRG_catalog_wRadioFlags_20260519", # LRG sample from Yu-Lin's paper
                'BGS': f"{self.path}/for_chad_20260519/BGS_catalog_wRadioFlags_20260519", # BGS sample for Boryana's paper
                'optical': f"{self.path}/for_chad_20260519/optical_catalog_wRadioFlags_20260519",  # Optical sample from Yun-Hsuin
                        }
            self.dfdata_raw = pd.read_csv(f"{sampfiles[self.sample]}.csv")  # import datafarme
            self.dfdata = self.dfdata_raw.copy()[self.dfdata_raw[f"valid_radio_{self.radio}"]==1]
        

    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None, logMsMin=None):
        self.get_cat()
        self.catdist('z', self.dfdata.z, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None):
        self.get_cat()
        if self.sample=='LRG':
            masses = np.log10(self.dfdata['Mstar'])
            
        elif self.sample=='BGS':
            masses = self.dfdata['logm']
    
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMs', masses, qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None):
        self.get_cat()
        if self.sample=='LRG':
            self.require(['masstype'])
            masses = np.log10(self.dfdata[self.masstype])
            
        elif self.sample=='BGS':
            masses = SHMRs.Gao2023(model='Auto').HSMR(self.dfdata['logm'])()
            
        elif self.sample=='optical':
            masses = np.log10(SHMRs.To2020(h=0.7).RHMR(self.dfdata['lum'])())
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMh', masses, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)


class Measurements(BaseMeasurement, Studies.Moore2026):
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
