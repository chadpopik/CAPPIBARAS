"""
The Atacama Cosmology Telescope: arcminute-resolution maps of 18 000 square degrees of the microwave sky from ACT 2008-2018 data combined with Planck

ui.adsabs.harvard.edu/abs/2020JCAP...12..046N
arxiv.org/pdf/2007.07290
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Naess2020")




"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2020JCAP...12..046N
    subs={}
    info={}

class Measurements(Study):  # ACT DR5 (Naess 2020, arxiv.org/abs/2007.07290)
    path = f"{DATA_PATH}/ACTDR5"  # Path to data downloaded from lambda.gsfc.nasa.gov/product/act/actpol_dr5_aux_prod_get.html
    # NERSC_path = "/global/cfs/projectdirs/act/data/act_dr5/s08s18_coadd/auxilliary"  # location of data in NERSC
    
    subs = {'freq': ['090', '150', '220']}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['freq'])

        self.beamfile = f"{self.path}/beams/act_planck_dr5.01_s08s18_f{self.freq}_daynight_beam.txt"  # Map beams transfer function: ells, B
        self.beam_ells, self.beam_data = np.genfromtxt(self.beamfile).T  # [ells, unitless]

        self.respfile = f"{self.path}/responses/act_planck_dr5.01_s08s18_AA_f{self.freq}_daynight_response_tsz.txt"  # Map-averaged response to tSZ: ell, I, dI, Q, dQ, U, dU
        self.resp_ells, self.resp_data = np.genfromtxt(self.respfile).T[0:2]


class MapData(Study):  # ACT DR5
    path = f"{DATA_PATH}/ACTDR5"  # Path to data downloaded from lambda.gsfc.nasa.gov/product/act/actpol_dr5_aux_prod_get.html
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
