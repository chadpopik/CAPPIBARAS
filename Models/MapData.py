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



datapath = "/global/homes/c/cpopik/Data"  # path to data


class BaseMapData:
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)



class ACTDR6(BaseMapData, Studies.Coulton2024):  # ACT DR6 component-separated maps
    path = f"{datapath}/ACTDR6"  # Path to data downloaded from portal.nersc.gov/project/act/dr6_nilc/ymaps_20230220/
    otherpath = "/global/cfs/projectdirs/act/data/act_dr6/dr6.02/beams/daytime_beams/"
    # NERSC_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220"  # path to data in NERSC
    subs={'freq': ['f150', 'f220','f090']}
    info={}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
        # beamdf = pd.loadtxt(f"{self.path}/beam_full_pa4_{self.freq}_daydeep_modes.txt")
        ybeamdf = pd.read_csv(f"{self.path}/ilc_beam.txt", sep=" ")
        self.beam_ells, self.beam_data = ybeamdf['#'].values, ybeamdf['ell'].values
        
        fwhm_arcmin = {'f150': 1.4, 'f220': 1.0, 'f090': 2.1}[self.freq]
        sigma = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
        self.beam_data = np.exp(-0.5 * self.beam_ells * (self.beam_ells + 1) * sigma**2)
        
        self.resp_ells, self.resp_data = None, None

    def get_beam(self):
        return self.beam_ells, self.beam_data
    

class Coulton2024(BaseMapData, Studies.Coulton2024):  # ACT DR6 component-separated maps
    path = f"{datapath}/ACTDR6"  # Path to data downloaded from portal.nersc.gov/project/act/dr6_nilc/ymaps_20230220/
    otherpath = "/global/cfs/projectdirs/act/data/act_dr6/dr6.02/beams/daytime_beams/"
    # NERSC_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220"  # path to data in NERSC
    subs={}
    info={}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        ybeamdf = pd.read_csv(f"{self.path}/ilc_beam.txt", sep=" ")
        self.beam_ells, self.beam_data = ybeamdf['#'].values, ybeamdf['ell'].values
        self.resp_ells, self.resp_data = None, None

    def get_beam(self):
        return self.beam_ells, self.beam_data


class Naess2020(BaseMapData, Studies.Naess2020):  # ACT DR5
    path = f"{datapath}/ACTDR5"  # Path to data downloaded from lambda.gsfc.nasa.gov/product/act/actpol_dr5_aux_prod_get.html
    # NERSC_path = "/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary"  # location of data in NERSC
    
    subs = {'freq': ['090', '150', '220']}
    info={}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
    def get_beam(self):
        self.require(['freq'])
        self.beamfile = f"{self.path}/beams/act_planck_dr5.01_s08s18_f{self.freq}_daynight_beam.txt"  # Map beams transfer function: ells, B
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]
        return self.beam_ells, self.beam_data
    
    def get_resp(self):
        self.require(['freq'])
        self.respfile = f"{self.path}/responses/act_planck_dr5.01_s08s18_AA_f{self.freq}_daynight_response_tsz.txt"  # Map-averaged response to tSZ: ell, I, dI, Q, dQ, U, dU
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]  # [ells, uk/y]
        return self.resp_ells, self.resp_data