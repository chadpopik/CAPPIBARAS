"""
Data from studies

"""

import os
from astropy.table import Table
import numpy as np
import astropy
import importlib
import astropy.cosmology
import astropy.units as u
import astropy.constants as c
import astropy.cosmology.units as cu
import pandas as pd
import h5py, json, tarfile
from scipy.interpolate import RegularGridInterpolator
# u.h70 = u.def_unit('h70', format={'latex': r'h_{70}'})

import Models.Studies as Studies
import Models.HaloModels as HaloModels
import Models.SHMRs as SHMRs

from config import DATA_PATH, STACKING_PATH

datapath = DATA_PATH

class BaseTargetData:
    # From a catalof, make histogrammed distributions of a given column
    def catdist(self, qName, qs, qMin=None, qMax=None, dq=None, qNum=None, densspace=None):
        # Try to set range with input values first, then with info dict values, then just from data
        qMin = qMin if qMin is not None else ((getattr(self, f'{qName}Min', None)*u.dimensionless_unscaled).value if getattr(self, f'{qName}Min', None) is not None else qs.min())
        qMax = qMax if qMax is not None else ((getattr(self, f'{qName}Max', None)*u.dimensionless_unscaled).value if getattr(self, f'{qName}Max', None) is not None else qs.max())
        # Pretty ranges, may use later
        # if qmin is None: qmin = np.floor(qs.min()/dq)*dq  # default min
        # if qmax is None: qmax = np.ceil(qs.max()/dq)*dq  # default max
        # Try to get array size from input size of spacing
        try: qNum = qNum if qNum is not None else np.int32((qMax-qMin)/dq)
        except: raise ValueError("Must specify array spacing or length")

        setattr(self, f'{qName}Bins', np.linspace(qMin, qMax, qNum))  # make linearly spaced bins
        setattr(self, qName, (getattr(self, f'{qName}Bins')[1:]+getattr(self, f'{qName}Bins')[:-1])/2)  # center of bins
        setattr(self, f'd{qName}', getattr(self, qName)[1]-getattr(self, qName)[0])  # width of the bins
        setattr(self, f'N_{qName}', np.histogram(qs, bins=getattr(self, f'{qName}Bins'))[0])  # number distribution
        setattr(self, f'dNd{qName}', getattr(self, f'N_{qName}')/getattr(self, f'd{qName}'))  # differential number distribution
        if densspace is not None:  # if given area/volume, use that for the density
            setattr(self, f'n_{qName}', getattr(self, f'N_{qName}')/densspace)  # number density distribution
            setattr(self, f'dnd{qName}', getattr(self, f'dNd{qName}')/densspace)  # differential number density distribution


    # TODO: make 2D dist from samples with two values
    # def make_N_q1_q2(self, q1s, dq1, q2s, dq2, q1min=None, q1max=None, q2min=None, q2max=None):  # make 2D dist from samples with two values
    #     N_q1, q1cents, q1bins = self.make_N_q(q1s, dq1, q1min, q1max)  # make 1D dist of value 1
    #     N_q2, q2cents, q2bins = self.make_N_q(q2s, dq2, q2min, q2max)  # make 1D dist of value 1
    #     N_q1_q2, _, _ = np.histogram2d(q1s, q2s, bins=[q1bins, q2bins])  # make 2D number dist from hist
    #     # dNdq1dq2 = N /dq1/u.dex /dq2/u.dex  # apply units [dex^-2]
    #     return N_q1_q2, N_q1, N_q2, q1cents, q2cents

    def distave(self, q, qdist, qint=None):
        qint = qint if qint is not None else q
        return np.trapz(q*qdist, qint)/np.trapz(qdist, qint)

    
    
    
class Schaan2021(BaseTargetData, Studies.Schaan2021):  # ACT DR5 maps stacked on SDSS BOSS DR10
    path = f"{datapath}/Schaan2021"  # path to data, shared by author
    subs = {
        'sample': ['cmass', 'lowz'],}
    
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
        self.bigdata_cmass = np.loadtxt(f'{datapath}/Schaan2021/catalog.txt')
          
    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None):
        self.catdist('z', self.bigdata_cmass[:, 2], qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None):
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMs', self.bigdata[:, 18], qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None):        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMh', self.bigdata[:, 20], qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)
        
    


class Zhou2023(BaseTargetData, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B"  # path to data from zenodo.org/records/8319955
    # rawpath = f'/global/cfs/projectdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/catalogs' # path to raw data from https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/
    # https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/

    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }

    def __init__(self, inputsdict={}, **inputvars):       
        self.setup(inputsdict | inputvars)

    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None):
        self.require(['zbin', 'sample', 'hemisphere'])

        sampstr = 'extended_' if self.sample=='ext' else ''
        rawcat = Table.read(f"{self.path}/dr9_{sampstr}lrg_pzbins.fits")
        
        # spec = importlib.util.spec_from_file_location("cuts", f"{self.path}/quality_cuts.py")
        # cuts = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(cuts)
        # self.zs_df = cuts.apply_cuts(rawcat).to_pandas()
        
        self.finalcat = rawcat.to_pandas()
        zscat = self.finalcat[self.finalcat.Z_PHOT_MEDIAN>=0]
        zscat = zscat
        
        self.catdist('z', zscat, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    # def get_dndz(self):
    #     samp = {'main':'main', 'ext':'extended'}[self.sample]
    #     cols = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get columns from first row
    #     zdf = pd.read_csv(f"{self.path}/{samp}_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", skiprows=1, names=cols)  # format into dataframe
    #     self.z = (zdf.zmin.values+zdf.zmax.values)/2
    #     pzstr = f'bin_{self.zbin[-1]}_{self.hemisphere}' if self.zbin!='all' else 'all'  # get name of column corresponding on bin
    #     self.dz = self.z[1]-self.z[0]
    #     self.dndz = zdf[pzstr].values//self.dz /u.deg**2/u.dex
    #     self.dNdz = self.dndz * (self.area*u.deg**2)
        

    

class RiedGuachalla2025(BaseTargetData, Studies.RiedGuachalla2025):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs
    path = f"{datapath}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass subsample
    }
    
    info = {
        'area': 4300 *u.deg**2,  # overlapping region of ACT and DESI [deg^2], F1 and III.B.p4
        'logMhMean': 13.4 *u.dimensionless_unscaled,  # Msun/littleh, estimated mean halo mass of LRG [Msun/h], III.B.p5
        'zMin': {# spectroscopic redshift bins, III.B.p6
            'all':0.4, 'z_1':0.4, 'z_2':0.6, 'z_3':0.8, 'z_4':0.95},
        'zMax': {# spectroscopic redshift bins, III.B.p6
            'all':1.1, 'z_1':0.6, 'z_2':0.8, 'z_3':0.95, 'z_4':1.1},
        'logMsMin': {# stellar mass bins [Msun], III.B.p7
            'all':10.5, 'mass_1':10.5, 'mass_2':11.2, 'mass_3':11.4, 'mass_4':11.6},
        'logMsMax': {# stellar mass bins [Msun], III.B.p7
            'all':12.40, 'mass_1':11.2, 'mass_2':11.4, 'mass_3':11.6, 'mass_4':12.5},
        'zMean': {# mean redshift, T2
            'all':0.74, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.76, 'mass_2':0.75, 'mass_3':0.71, 'mass_4':0.69},
        'zMed': {# median redshift, T2
            'all':0.75, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.79, 'mass_2':0.76, 'mass_3':0.70, 'mass_4':0.67},
        'MsMean': {'units': 1e11*u.Msun, # mean stellar mass , T2
            'all':2.2, 'z_1':2.4, 'z_2':2.3, 'z_3':2.0, 'z_4':2.1, 'mass_1':1.2, 'mass_2':2.0, 'mass_3':3.0, 'mass_4':5.1},
        'NGal': {# number of galaxies, T2
            'all':825283, 'z_1':195877, 'z_2':235620, 'z_3':235620, 'z_4':96346, 'mass_1':244932, 'mass_2':320914, 'mass_3':194037, 'mass_4':53997},
    }
    
    # Assume z/m bins have same m/z limits as all
    for b in [1, 2, 3, 4]:
        info['zMin'][f'mass_{b}'] = info['zMin']['all']
        info['zMax'][f'mass_{b}'] = info['zMax']['all']
        info['logMsMin'][f'z_{b}'] = info['logMsMin']['all']
        info['logMsMax'][f'z_{b}'] = info['logMsMax']['all']
    
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_zs = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # load redshifts of all galaxies in catalog
        if self.bin[0]=='z': self.cat_zs = self.allcat_zs[f'{self.bin}']  # if zbin is specified, select for that subsample
        else: self.cat_zs = np.concatenate([self.allcat_zs[f'z_{i}'] for i in range(1, 5)])  # otherwise, get all of them

        self.catdist('z', self.cat_zs, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
        if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
        else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.zMax)-halomodel.Vcom(self.zMin))/(1+self.zMean)**3
        
        self.catdist('logMs', self.cat_logMs, qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
        if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
        else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
        # Get function to convert stellar masses to halo masses
        self.cat_logMh = SHMRs.Gao2023({'model':'Psat'}).HSMR(self.cat_logMs)()  # Use SHMR of Gao 2023 
        # TODO: 1. is the Psat model the best to use?, 2. think this converts to virial mass
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.zMax)-halomodel.Vcom(self.zMin))/(1+self.zMean)**3

        self.catdist('logMh', self.cat_logMh, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)


    # TODO: add 2D z/M distribution?
    
    
class Hadzhiyska2026(BaseTargetData, Studies.Hadzhiyska2026):
    path = f"{datapath}/Hadzhiyska2026"  # location of data, provided by Jenna

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        file = "BGS_BRIGHT_full_noveto_vac_marvin_BGS_ACT_DR6_fixedTh2.10_delta_Ts"
        self.dfdata = pd.read_csv(f"{self.path}/{file}.csv")

    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None, logMsMin=None):
        zscat = self.dfdata.z if logMsMin is None else self.dfdata[self.dfdata.logm>logMsMin].z  # for recreating the plot
        self.catdist('z', zscat, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None):
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMs', self.dfdata['logm'], qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None):
        catlogMhs = SHMRs.Gao2023(model='Auto').HSMR(self.dfdata['logm'])()
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

        self.catdist('logMh', catlogMhs, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)


class Moore2026(BaseTargetData, Studies.Moore2026):
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
    'LMin': {'units': 1e10 *u.Lsun,
        'LRG': {'L36': 3.58, 'L48': 4.78, 'L60': 5.98, 'L79': 7.86, 'L98': 9.80, 'L36D': 3.58, 'L48D': 4.78, 'L60D': 5.98, 'L79D': 7.86}},
    'LMax': {'units': 1e10 *u.Lsun,
        'LRG': {'L36D': 4.78, 'L48D': 5.98, 'L60D': 7.86, 'L79D': 9.80}},
    'MhMin': {'units': 1e13 * u.Msun,   # virial mass, 1e13 Msun
        'LRG': {'L36': 0.38, 'L48': 0.63, 'L60': 0.96, 'L79': 1.64, 'L98': 2.59, 'L36D': 0.38, 'L48D': 0.63, 'L60D': 0.96, 'L79D': 1.64}},
    'MhMax': {'units': 1e13 *u.Msun,
        'LRG': {'L36D': 0.63, 'L48D': 0.96, 'L60D': 1.64, 'L79D': 2.59}},
    'MsMean': {'units': 1e11 *u.Msun,
        'LRG': {'L36': 2.07, 'L48': 2.34, 'L60': 2.70, 'L79': 3.35, 'L98': 4.10, 'L36D': 1.26, 'L48D': 1.61, 'L60D': 2.04, 'L79D': 2.61}},
    'LMean': {'units': 1e10 *u.Lsun,
        'LRG': {'L36': 6.89, 'L48': 7.78, 'L60': 9.00, 'L79': 11.18, 'L98': 13.67, 'L36D': 4.21, 'L48D': 5.36, 'L60D': 6.82, 'L79D': 8.71}},
    'zMean': {
        'LRG': {'L36': 0.76, 'L48': 0.78, 'L60': 0.80, 'L79': 0.84, 'L98': 0.88, 'L36D': 0.70, 'L48D': 0.72, 'L60D': 0.76, 'L79D': 0.81},
         'BGS': {'M10': 0.25, 'M1025': 0.27, 'M105': 0.28, 'M1075': 0.30, 'M11': 0.33, 'M1125': 0.36},
         'optical': {'Full': 0.48, 'L10': 0.48, 'L20': 0.47}},
    'N': {
        'LRG': {'L36': 913286, 'L48': 685766, 'L60': 456803, 'L79': 228650, 'L98': 114024, 'L36D': 227520, 'L48D': 228963, 'L60D': 228153, 'L79D': 114626},
        'BGS': {'M10': 1610381, 'M1025': 1379230, 'M105': 1084339, 'M1075': 744562, 'M11': 421150, 'M1125': 178206},
        'optical': {'Full': 220698, 'L10': 80447, 'L20': 22320}},
    'Nf150': {
        'LRG': {'L36': 957095, 'L48': 718542, 'L60': 478585, 'L79': 239567, 'L98': 119353, 'L36D': 238553, 'L48D': 239957, 'L60D': 239018, 'L79D': 120214}},
    'logMsMin': {
        'BGS': {'M10': 10.00, 'M1025': 10.25, 'M105': 10.50, 'M1075': 10.75, 'M11': 11.00, 'M1125': 11.25}},
    'logMsMean': {
        'BGS': {'M10': 10.72, 'M1025': 10.82, 'M105': 10.94, 'M1075': 11.08, 'M11': 11.25, 'M1125': 11.42}},
    'lambdaMin': {
        'optical': {'Full': None, 'L10': 10, 'L20': 20}},
    'lambdaMean': {
        'optical': {'Full': 7.04, 'L10': 11.81, 'L20': 20.42}},
    'logMhMean': {# M200m, h^-1 Msun
        'optical': {'Full': 13.45, 'L10': 13.76, 'L20': 14.08}},
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)   
        
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
        
    # def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None):
    #     catlogMhs = SHMRs.Gao2023(model='Auto').HSMR(self.dfdata['logm'])()
        
    #     vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.dfdata.z.max())-halomodel.Vcom(self.dfdata.z.min()))/(1+self.dfdata.z.mean())**3

    #     self.catdist('logMh', catlogMhs, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)







class Liu2025(BaseTargetData, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs
    path = f"{datapath}/Liu2025"  # path to data from zenodo.org/records/14706729
    subs = {
        'zbin' : ['all', 'z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        # 'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        # 'freq' : ['090', '150'],  # y map frequency (DR5)
    }
    
    info = {
        'area': 7326 *u.deg**2,  # area of ACT DESI overlap, Fig 1
        'logMhMean': { # rough mean halo mass taken from Yuan 2023, III.A.p5
            'z1':13.40, 'z2':13.40, 'z3':13.24, 'z4':13.24},  
        'zMean': {  # mean redshift, T1
            'z1':0.470, 'z2':0.628, '3':0.791, 'z4':0.924},
        'nGal': {'units': u.deg**-2, #  mean number density, T1
            'z1':81.9, 'z2':148.1, 'z3':162.4, 'z4':148.3},
        'NGal_unmasked': { # objects in catalog, T1
            'z1':1118496, 'z2':2031303, 'z3':2240982, 'z4':2049158},
        'NGal': { # objects in ACT/DESI overlap, T1
            'z1':332280, 'z2':608100, 'z3':671738, 'z4':615543},
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

    def make_zdist(self, dz=None, zMin=None, zMax=None, **kwargs):
        self.require(['zbin'])
        zstr = f'bin_{self.zbin[-1]}' if self.zbin!='all' else 'all'
        self.n_df = self.zdf[f'{zstr}_combined'].values /u.deg**2
        self.dndz_df = self.n_df /(self.zs_df[1]-self.zs_df[0])
        zmin = zMin if zMin is not None else self.zs_df.min()
        zmax = zMax if zMax is not None else self.zs_df.max()
        self.dz = dz if dz is not None else self.zs_df[1]-self.zs_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)
        self.dndz = np.interp(self.z, self.zs_df, self.dndz_df)
        
        self.dNdz = self.dndz*self.area
        self.n_z = self.dndz*self.dz
        self.N_z = self.dNdz*self.dz

    def make_Mhdist(self, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None, halomodel=None):
        self.logMhMin = 12
        self.logMhMax = 14
        self.catdist('logMh', np.zeros([100]), qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=1)
        









class Kusiak2022(BaseTargetData, Studies.Kusiak2022):  # unWISE galaxies and Planck lensing
    path = f"{datapath}/Kusiak2022"  # path to data, provided by author
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
        
        self.dNdz = self.dNdz_norm*self.area*self.ndens
        self.dndz = self.dNdz_norm*self.ndens
        self.N_z = self.dNdz_norm*self.area*self.ndens*self.dz
        self.n_z = self.dNdz_norm*self.ndens*self.dz
        
        

class White2022(BaseTargetData, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }
    
    info = {
        'area': 18000 *u.deg**2,  # area of DESI survey, [deg^2], Figure 1
        # info, Table 1
        'zMean': {# mean redshift
            'z1':0.47, 'z2':0.63, 'z3':0.79, 'z4':0.92},
        'ndens': {'units':u.deg**-2, # galaxy number desnity
            'z1':83, 'z2':149, 'z3':162, 'z4':149},
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
        self.dNdz = self.N_z/self.dz
        self.dndz = self.n_z/self.dz
        self.dlnNdz = np.log(self.N_z.value)/self.dz
        
        # TODO: think these units of density are weird, check
        

        
    




    
    
    
    
    
    
    
    
    
    

    
    



    
    
    
    
    
    
    
    
    
    







        
        
        
        
        


class Popik2026(BaseTargetData, Studies.Popik2026):  # TODO: In progress
    path = f"{STACKING_PATH}/Results"
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
        
        self.make_zdists

    def make_zMsdists(self, dz=None, zMin=None, zMax=None, dlogMs=None, logMsMin=None, logMsMax=None):
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z_df = (zbins[1:]+zbins[:-1])/2
        self.dz_df = self.z_df[1]-self.z_df[0]
        
        # Read the plot data from the files
        self.logMs_df = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.n_logMs_z_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]
        
        self.dndzdlogMs_df = self.n_logMs_z_h3 *self.h**3 /u.Mpc**3 /self.dz_df/u.dex
        
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
        self.dNdzdlogMs = dNinterp(np.column_stack([zgrid.ravel(), logMsgrid.ravel()])).reshape(len(self.z), len(self.logMs)) / u.dex
        
        self.dNdogMs_z = self.dNdzdlogMs *self.dz
        self.N_z = np.trapz(self.dNdogMs_z, self.logMs)
        self.n_z = self.N_z / self.area
        self.dNdz = self.N_z / self.dz
        self.dndz = self.dNdz / self.area
        
        self.N_z_logMs = self.dNdogMs_z *self.dlogMs*u.dex
        

    # def dndlogmstar(self, hh=0.71, **kwargs):  # Add a h^3 factor to convert from (Mpc/h)^-3 to Mpc^-3
    #     return self.dndlogmstar_h3*hh**3
    
    # def dNdz(self, **cosmopars):
    #     return np.trapz(self.dndlogmstar(**cosmopars), self.logmstar)*self.volumes(**cosmopars)/(self.z[1]-self.z[0])


class Kou2023(BaseTargetData, Studies.Kou2023):
    path = f"{datapath}/Kou2023"  # path to data, taken from plots using webplotdigitizer
    subs = {'mbin':['M1', "M2", "M3", "M4"]}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['mbin'])









class Amodeo2021(BaseTargetData, Studies.Amodeo2021):  # Inference on BOSS DR10 stacked ACT DR5 (arxiv.org/abs/2009.05558)
    path = f"{datapath}/Amodeo2021/"  # path to data, taken from plots using webplotdigitizer and various repos
    subs = {'prof': ['Amodeo', 'Battaglia', 'TNG']}
    
    info = { # galaxy catalog detailed, II.Ap1/Ap4/Figure2
        'MsMean': 3e11 *u.Msun, 'MhMean': 3.3e13 *u.Msun, # mean stellar and halo mass
        'zMin':0.4, 'zMax':0.7, 'zMed': 0.55,  # min/max/median redshift
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def get_dNdlogmhalo(self, dlogmhalo=0.05, dlogmstar=0.05):
        massdist = np.loadtxt(f'{self.path}/mass_distrib.txt')        
        self.dNdlogMh, self.logMh, Mhalobins = self.dNdq_cat(np.log10(massdist), dlogmhalo)  # Get 1D hist for z

        massdiststar = SHMRs.Kravstov2018({'mdef':'200c', 'scatter':'S'}).HSMR(np.log10(massdist))()  # steal default SHMR
        self.dNdlogMs, self.logMs, Msbins = self.dNdq_cat(massdiststar, dlogmstar)  # Get 1D hist for z


class More2015(BaseTargetData, Studies.More2015):
    path = None
    subs = {''}




# class SDSSBOSS(BaseTargetData, Studies.Ahn2013Alam2015):  # (Ahn+ 2013, arxiv.org/abs/1307.7735, Alam+ 2015, https://arxiv.org/abs/1501.00963)
#     path = '/global/cfs/projectdirs/sdss/data/sdss'   # Path of the data in NERSC
#     if not os.path.isdir(path): # Path through a URL if not in NERSC
#         path = 'https://data.sdss.org/sas/'
#     subs = {
#         'DR': ['DR10', 'DR12'],
#         'galaxy': ['CMASS', 'LOWZ'],  # Galaxy sample
#         'group': ['portsmouth', 'wisconsin', 'granada'],  # Group models
#         'IMF': ['krou', 'salp'],  # Kroupe or Salpeter Initial Mass Function
#         'template': ['starforming', 'passive'],
#         'pop': ['bc03', 'm11'], # Bruzual-Charlot or Maraston population
#         'time': ['earlyform', 'wideform'],  # early or extended SF
#         'dust': ['dust', 'nodust'],
#     }

#     def __init__(self, inputsdict, **inputvars):
#         self.setup(inputsdict | inputvars)
#         self.require(['DR'])
        
#         self.get_catalog()
        
#     def get_catalog(self):  # import catalog
#         self.require(['group'])  # base info
#         vers = {'DR10':['v1_0', '5_12'], 'DR12':['v1_1', '7_0']}[self.DR]
#         self.path = f"{self.path}/dr{self.DR[2:]}/boss/spectro/redux/galaxy/{vers[0]}"  # locate folder
            
#         if self.group=='portsmouth': 
#             self.require(['template', 'IMF'])
#             fname, mcolname = f"{self.group}_stellarmass_{self.template}_{self.IMF}-v5_{vers[1]}.fits.gz", 'LOGMASS'
#         elif self.group=='wisconsin': 
#             self.require(['pop'])
#             fname, mcolname = f"{self.group}_pca_{self.pop}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'
#         elif self.group=='granada': 
#             self.require(['template', 'time', 'dust'])
#             fname, mcolname = f"{self.group}_fsps_{self.IMF}_{self.time}_{self.dust}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'

#         # Fetch the data and rename the mass column, sdss4.org/dr17/spectro/galaxy_portsmouth/
#         self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

#         # Select the correct galaxy sample using the bitmasks in sdss3.org/dr10/algorithms/bitmask_boss_target1.php, sdss3.org/dr10/algorithms/bitmasks.php, sdss4.org/dr17/algorithms/bitmasks/, skyserver.sdss.org/dr19/MoreTools/browser/
#         decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
#         self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
#         self.bitmasks = {'CMASS':7, 'LOWZ':0}

#     def make_dNdz(self, dz=0.1, zMin=None, zMax=None, **kwargs):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         self.N_z, self.z, zbins = self.make_N_q(dfdata.Z, dz, zMin, zMax)
#         self.dz = dz
#         self.dNdz = self.N_z / self.dz/u.dex
#         self.dndz = self.dNdz/ self.area/u.deg**2
    
#     def get_dNdlogMs(self, dlogMs=0.05, logMsmin=None, logMsmax=None):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         dNdlogMs, logMs, logMsbins = self.make_N_q(dfdata.LOGM, dlogMs, logMsmin, logMsmax)
#         return dNdlogMs, logMs, dlogMs
        
#     def get_dNdlogMs(self, dlogMs=0.05, dz=0.05, zmin=None, zmax=None, logMsmin=None, logMsmax=None):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         dNdlogMsdz, dNdlogMs, dNdz, logMs, z = self.make_N_q1_q2(dfdata.LOGM, dlogMs, self.dfdata.Z, dz)
#         dNdlogMs2D = dNdlogMsdz*dz*u.dex
#         return dNdlogMs2D, z, logMs, dz, dlogMs








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
    
    
    
    
        # TODO: handle this stuff
        # norm_fac = self.dNdz[:,None]/np.trapz(self.dNdlogmhalo, self.logmhalos)*(self.zs[1]-self.zs[0])  # ensures same zcount as dNdz
        # self.dndlogmhalo = self.dNdlogmhalo*norm_fac/self.volumes()[:, None]  # create number dist [dex^-1]