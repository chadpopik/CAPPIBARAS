"""
Collection of Stellar Halo Mass Relations models and paramterizations for various studies/samples.

"""

import numpy as np
import Models.Studies as Studies

class BASESHMR:
    # SHMR form from Behroozi 2013 (Behroozi+ 2013, arxiv.org/abs/1207.6105)
    def Behroozi(self, logMh, logM1, logeps, alpha, delta, gamma):
        Mh, M1, eps = 10**logMh, 10**logM1, 10**logeps
        f = lambda x : -np.log10(10**(alpha*x)+1) + delta*(np.log10(1+np.exp(x)))**gamma/(1+np.exp(10**(-x)))
        Ms = 10**( np.log10(eps*M1) + f(np.log10(Mh/M1)) - f(0) )
        return np.log10(Ms)

    # Double Power Law SHMR form (Moster+ 2012, arxiv.org/abs/1205.5807)
    def DoublePowerLaw(self, logMh, logM1, N, beta, gamma):
        Mh, M1 = 10**logMh, 10**logM1
        Ms = 2*N/((Mh/M1)**(-beta) + (Mh/M1)**(gamma))
        return np.log10(Ms)

    # Get halo mass from stellar mass using interpolation
    def HSMR(self, logMs):
        logMhs = np.linspace(10, 20, 1000)  # Should cover the range of reasonable halo masses
        func = lambda p: np.interp(logMs, self.SHMR(logMhs)(self.p0 | p), logMhs)
        return lambda p={}: func(self.p0 | p)


class Kravtsov2018(BASESHMR, Studies.Kravtsov2018):  # SDSS DR8 & G13 
    models = {
            'model': ['B13', 'PL'],
            'type': ['BGC', 'sat', 'tot'],
            'data': ['K18', 'K18G13'],
            'mdef': ["200c", "500c", "200m", "vir"],
            'scatter': ['B', 'S']}
    params={
        # best fit SHMR params, Table 3
        "logM1": {
            "B": {"200c": 11.39, "500c": 11.32, "200m": 11.45, "vir": 11.43},
            "S": {"200c": 11.35, "500c": 11.28, "200m": 11.41, "vir": 11.39},},
        "logeps": {
            "B": {"200c": -1.618, "500c": -1.527, "200m": -1.702, "vir": -1.663},
            "S": {"200c": -1.642, "500c": -1.556, "200m": -1.720, "vir": -1.685},},
        "alpha": {
            "B": {"200c": 1.795, "500c": 1.856, "200m": 1.736, "vir": 1.750},
            "S": {"200c": 1.779, "500c": 1.835, "200m": 1.727, "vir": 1.740},},
        "delta": {
            "B": {"200c": 4.345, "500c": 4.376, "200m": 4.273, "vir": 4.290},
            "S": {"200c": 4.394, "500c": 4.437, "200m": 4.305, "vir": 4.335},},
        "gamma": {
            "B": {"200c": 0.619, "500c": 0.644, "200m": 0.613, "vir": 0.595},
            "S": {"200c": 0.547, "500c": 0.567, "200m": 0.544, "vir": 0.531},},
        "slope": {
            "BCG": {"K18": 0.39, "K18G13": 0.33},
            "sat": {"K18": 0.87, "K18G13": 0.75},
            "tot": {"K18": 0.69, "K18G13": 0.59}},
        "norm": {
            "BCG": {"K18": 12.15, "K18G13": 12.24},
            "sat": {"K18": 12.42, "K18G13": 12.52},
            "tot": {"K18": 12.63, "K18G13": 12.71}},
        "scat": {
            "BCG": {"K18": 0.21, "K18G13": 0.17},
            "sat": {"K18": 0.10, "K18G13": 0.10},
            "tot": {"K18": 0.09, "K18G13": 0.11}},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        self.require(['model'])
        if self.model=='B13': self.SHMR = self.SHMR_B13
        elif self.model=='PL': self.SHMR = self.SHMR_PL
        
    def SHMR_PL(self, logMh):  # Eq A3/A4
        self.require(['type', 'data'])
        func = lambda p: p['slope']*(logMh-14.5)-p['norm']
        return lambda p={}: func(self.p0 | p)

    def SHMR_B13(self, logMh):  # Eq A3/A4
        self.require(['mdef', 'scatter'])
        func = lambda p: self.Behroozi(logMh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
        return lambda p={}: func(self.p0 | p)
    


class Xu2023(BASESHMR, Studies.Xu2023):  # SDSS DR7 Main and SDSSIII BOSS DR12 LOWZ & CMASS (arxiv.org/abs/2211.02665)
    models = {'sample': ['Main', 'LOWZ', 'CMASS'],  # galaxy sample
            'form': ['BP13', 'DP'],  # form of SHMR
            }
    params = {  # best-fit SHMR parameters
        "logM0": {  # Msun/h
            "BP13": {"Main": 11.338, "LOWZ": 11.359, "CMASS": 11.509},
            "DP":   {"Main": 11.732, "LOWZ": 11.579, "CMASS": 11.624}},
        "alpha": {  # slope of high mass end of SHMR
            "BP13": {"Main": 0.484, "LOWZ": 0.623, "CMASS": 0.740},
            "DP":   {"Main": 0.299, "LOWZ": 0.429, "CMASS": 0.466}},
        "delta": {
            "BP13": {"Main": 3.041, "LOWZ": 3.248, "CMASS": 2.964}},
        "beta": {  # slope of low mass end of SHMR
            "BP13": {"Main": 1.632, "LOWZ": 1.702, "CMASS": 2.094},
            "DP":   {"Main": 1.917, "LOWZ": 2.215, "CMASS": 2.513}},
        "logeps": {  # TODO: check units on this
            "BP13": {"Main": -1.545, "LOWZ": -1.598, "CMASS": -1.565}},
        "logk": {  # TODO: check units on this
            "DP":   {"Main": 10.303, "LOWZ": 10.105, "CMASS": 10.133}},
        "sigma": {  # width of gaussian function that scatter logMs at a given Macc, TODO: implement
            "BP13": {"Main": 0.237, "LOWZ": 0.190, "CMASS": 0.190},
            "DP":   {"Main": 0.233, "LOWZ": 0.201, "CMASS": 0.192}},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def SHMR(self, logMh):
        self.require(['form', 'sample'])
        if self.form=='BP13': return self.SHMR_BP13(logMh)
        elif self.form=='DP': return self.SHMR_DP(logMh)

    def SHMR_BP13(self, logMh):
        func = lambda p: self.Behroozi(logMh-np.log10(self.h), logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        return lambda p={}: func(self.p0 | p)

    def SHMR_DP(self, logMh):
        func = lambda p: self.DoublePowerLaw(logMh-np.log10(self.h), logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


class Gao2023(BASESHMR, Studies.Gao2023):  # DESI 1% (arxiv.org/abs/2306.06317)
    models = {'model':["Auto", "Cross", "Psat"],}
    params = {
        # best-fit SHMR parameters, Table 3
        'logM0': {'Auto': 11.56, 'Cross': 12.14, 'Psat': 12.07},  # divides the slopes
        'alpha': {'Auto': 0.43,  'Cross': 0.37,  'Psat': 0.37},  # slope
        'beta': {'Auto': 2.72,  'Cross': 2.27,  'Psat': 2.61},  # slope
        'logk': {'Auto': 10.11, 'Cross': 10.40, 'Psat': 10.36},  # normalization constant
        'sigma': {'Auto': 0.18,  'Cross': 0.21,  'Psat': 0.21},  # scatter, TODO: what to do with this?
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def SHMR(self, logMh):
        self.require(['model'])
        func = lambda p: self.DoublePowerLaw(logMh-np.log10(self.h), logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)

