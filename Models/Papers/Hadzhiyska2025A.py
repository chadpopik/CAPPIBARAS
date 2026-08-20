"""
Evidence for large baryonic feedback at low and intermediate redshifts from kinematic Sunyaev-Zel'dovich observations with ACT and DESI photometric galaxies

ui.adsabs.harvard.edu/abs/2025PhRvD.112h3509H
arxiv.org/pdf/2407.07152
"""


from config import *
from Models.Papers.Figures.PlotsTables import ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Hadzhiyska2025A")



class ParamsTable(ParamTable):  # best fit HOD parameters
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)




"""Old implementation I'm phasing out"""

from Models.Studies import BaseStudy
class Study(BaseStudy):  # arxiv.org/abs/2407.07152
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],}
    info={}
    
#     # info = {'ngal': {'ext_DR9_z1': 963631, 'ext_DR9_z2': 1658313, 'ext_DR10_z3': 1951646, 'ext_DR10_z4':1690171, 'ext_all':6850072},
#     # # TODO: come back to this
#     # }
#     info = {'name':'Hadzhiyska 2025'}

class Measurements(Study):  # Stacked kSZ measurement of ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
    path = f"{DATA_PATH}/Hadzhiyska2024"  # Path to data from zenodo.org/records/12633573
    subs = {
        'zbin': ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'DR':['all', 'DR9', 'DR10'],
        'sample': ['main', 'extended', 'all'],
        'zoutcut': ['nocut', 'cut'],
        'corr': ['corrected', 'uncorrected'],
    }

    def __init__(self, inputsdict={}, **inputvars):
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
        filename = f"{self.path}/Fig1_Fig8_{samplestr}dr10_allfoot_perbin_{zstr}dr6_{corrstr}_pzbin{self.zbin[-1]}.npz"

        self.TkSZ_data = np.load(filename)['prof'] *u.uK*u.arcmin**2
        self.TkSZ_cov = np.load(filename)['cov'] *(u.uK*u.arcmin**2)**2
        self.TkSZ_err = np.diag(self.TkSZ_cov)**0.5

        
from Models.TargetData import BaseTargetData
class TargetData(BaseTargetData, Study):  # Stacked kSZ measurement of ACT DR6 and DESI LRGs LIS DR9/10 (arxiv.org/abs/2407.07152)
    path = f"{DATA_PATH}/Hadzhiyska2024"  # Path to data from zenodo.org/records/12633573
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

