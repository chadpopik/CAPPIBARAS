"""
Collection of Stellar Halo Mass Relations models and paramterizations for various studies/samples.
"""

import numpy as np

class BASESHMR:
    # Check model specifications are in list and assign default parameters
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:  # 
                setattr(self, mname, spefs[mname])
        self.p0 = {param: self.params[param][self.samples.index(self.sample)] for param in self.params.keys()}

    # SHMR form from Behroozi 2013 (arxiv.org/abs/1207.6105)
    def Behroozi(self, logMh, logM1, logeps, alpha, delta, gamma):
        Mh, M1, eps = 10**logMh, 10**logM1, 10**logeps
        f = lambda x : -np.log10(10**(alpha*x)+1) + delta*(np.log10(1+np.exp(x)))**gamma/(1+np.exp(10**(-x)))
        Ms = 10**( np.log10(eps*M1) + f(np.log10(Mh/M1)) - f(0) )
        return np.log10(Ms)
    
    # Double Power Law SHMR form (arxiv.org/abs/1205.5807)
    def DoublePowerLaw(self, logMh, logM1, N, beta, gamma):
        Mh, M1 = 10**logMh, 10**logM1
        Ms = 2*N/((Mh/M1)**(-beta) + (Mh/M1)**(gamma))
        return np.log10(Ms)
    
    # Get halo mass from stellar mass using interpolation
    def SHMR(self, logMs):
        logMhs = np.linspace(10, 20, 1000)  # Should cover the range of reasonable halo masses
        func = lambda p: np.interp(logMs, self.HSMR(logMhs, self.p0 | p), logMhs)
        return lambda p={}: func(self.p0 | p)
    

class Xu2023(BASESHMR):  # SDSS Main DR7, CMASS & LOWZ DR12 (arxiv.org/abs/2211.02665)
    info = {'mdef': 'vir',  # virial mass of the halo at the time when the galaxy was last the central dominant object
    }
    samples = ["Main_BP13", "LOWZ_BP13", "CMASS_BP13", "Main_DP", "LOWZ_DP", "CMASS_DP"]
    params = {
        'logM0': [11.338, 11.359, 11.509, 11.732, 11.579, 11.624],
        'alpha': [0.484, 0.623, 0.740, 0.299, 0.429, 0.466],
        'delta': [3.041, 3.248, 2.964, None, None, None],
        'beta': [1.632, 1.702, 2.094, 1.917, 2.215, 2.513],
        'logeps': [-1.545, -1.598, -1.565, None, None, None],
        'logk': [None, None, None, 10.303, 10.105, 10.133],
        'sigma': [0.237, 0.190, 0.190, 0.233, 0.201, 0.192]
        }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        self.form = self.sample.split('_')[-1]

    def HSMR(self, logMh):
        if self.form=='BP13':
            func = lambda p: self.Behroozi(logMh, logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        elif self.form=='DP':
            func = lambda p: self.DoublePowerLaw(logMh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


# DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)
class Gao2023(BASESHMR):
    info = {'mdef':'vir',  # Current Virial Mass
            }
    samples = ["ELG_Auto", "ELG_Cross", "Psat_Mh"]
    params = {
        'logM0': [11.56, 12.14, 12.07],
        'alpha': [0.43, 0.37, 0.37],
        'beta': [2.72, 2.27, 2.61],
        'logk': [10.11, 10.40, 10.36],
        'sigma': [0.18, 0.21, 0.21]}
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])

    def HSMR(self, logMh):
        func = lambda p: self.DoublePowerLaw(logMh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


# SDSS DR8 (arxiv.org/abs/1401.7329)
class Kravstov2014(BASESHMR):
    info = {}
    samples = ["M200c", "M200c_scatter", "M500c", "M500c_scatter", "M200m", "M200m_scatter", "Mvir", "Mvir_scatter"]
    params = {
        "logM1": [11.39, 11.35, 11.32, 11.28, 11.45, 11.41, 11.43, 11.39],
        "logeps": [-1.618, -1.642, -1.527, -1.556, -1.702, -1.720, -1.663, -1.685],
        "alpha": [1.795, 1.779, 1.856, 1.835, 1.736, 1.727, 1.750, 1.740],
        "delta": [4.345, 4.394, 4.376, 4.437, 4.273, 4.305, 4.290, 4.335],
        "gamma": [0.619, 0.547, 0.644, 0.567, 0.613, 0.544, 0.595, 0.531]}

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        self.info['mdef'] = self.sample.split('_')[0][1:]

    def HSMR(self, logMh):
        func = lambda p: self.Behroozi(logMh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
        return lambda p={}: func(self.p0 | p)