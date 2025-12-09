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
    def SHMR(self, logMs):
        logMhs = np.linspace(10, 20, 1000)  # Should cover the range of reasonable halo masses
        func = lambda p: np.interp(logMs, self.HSMR(logMhs)(self.p0 | p), logMhs)
        return lambda p={}: func(self.p0 | p)


class BOSS_DR12(BASESHMR, Studies.Xu2023):  # SDSS DR7 Main and SDSSIII BOSS DR12 LOWZ & CMASS (arxiv.org/abs/2211.02665)
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

    def HSMR(self, logMh):
        if self.form=='BP13':
            func = lambda p: self.Behroozi(logMh, logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        elif self.form=='DP':
            func = lambda p: self.DoublePowerLaw(logMh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


class DESI_1P(BASESHMR, Studies.Gao2023):  # DESI 1% (arxiv.org/abs/2306.06317)
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

    def HSMR(self, logMh):
        func = lambda p: self.DoublePowerLaw(logMh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


class SDSS_DR8(BASESHMR, Studies.Kravstov2018):  # SDSS DR8 & G13 (arxiv.org/abs/1401.7329)
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

    def HSMR(self, logMh):
        func = lambda p: self.Behroozi(logMh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
        return lambda p={}: func(self.p0 | p)