"""
Evidence for large baryonic feedback at low and intermediate redshifts from kinematic Sunyaev-Zel'dovich observations with ACT and DESI photometric galaxies

ui.adsabs.harvard.edu/abs/2025PhRvD.112h3509H
arxiv.org/pdf/2407.07152
"""

import os
from Models.Papers.PlotsTables import ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Studies(BaseStudy):  # Missing baryons recovered: a measurement of the gas fraction in galaxies and groups with thekinematic Sunyaev-Zel’dovich effect and CMB lensing, ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
    subs = {
    }

    info = {'h': 0.6736, 'MassDef': 'vir'
    }


class ParamsTable(ParamTable):  # best fit HOD parameters
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)


class HODs(BaseHOD, Studies.Hadzhiyska2025A):  #
    models = {'sample': ['Main_z1', 'Main_z2', 'Main_z3', 'Main_z4', 'Main_all', 'Ext_z1', 'Ext_z2', 'Ext_z3', 'Ext_z4', 'Ext_all', 'BGS'],
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = ParamsTable().getparams(sample=self.sample).to_dict()

    def Ncen(self, logM):  # Eq 2
        logM = logM-np.log10(self.h)
        func = lambda p: Zheng2005().Nc(logM, logMmin=p['logMcut'], sigmalogM=2*p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 3
        logM = logM-np.log10(self.h)
        func = lambda p: Zheng2005().Ns(logM, M0=p['kappa']*10**p['logMcut'], M1=10**p['logM1'], alpha=p['alpha']) * self.Ncen(logM)(p)
        return lambda p={}: func(self.p0 | p)
