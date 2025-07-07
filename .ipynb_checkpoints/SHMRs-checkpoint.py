"""
Collection of Stellar Halo Mass Relations models and paramterizations for various studies/samples.
Classes should contain functions for converting host halo masses (in m200c) to galaxy stellar masses (HSMR) and the parameters values outlined in the papers, as well as the reverse conversion (SHMR) constructed through interpolation. 
Functions should also take an input dictionary to specify custom values for parameters, defaulting to the base values if not given. 
To more easily compare parameter values between models, they should be built off the Base Model functions.


# TODO 1: For all, check to make sure M_sun (with no h) is used for stellar masses, m200c is used for halo masses, and if it's not, figure out how to convert to it.
"""

from Basics import *



class BASESHMR:
    def checksample(self, sample):  # Set default parameterization from input sample, checking for validity
        if sample in self.samples: 
            self.sample = sample
            self.p0 = {param: self.params[param][self.samples.index(self.sample)] for param in self.params.keys()} 
        else: 
            raise NameError(f"Sample {sample} doesn't exist, choose from available samples: {self.samples}")

    def Behroozi(self, Mh, logM1, logeps, alpha, delta, gamma):  # (arxiv.org/abs/1207.6105)  TODO 1
        M1, eps = 10**logM1, 10**logeps
        f = lambda x : -np.log10(10**(alpha*x)+1) + delta*(np.log10(1+np.exp(x)))**gamma/(1+np.exp(10**(-x)))
        return 10**( np.log10(eps*M1) + f(np.log10(Mh/M1)) - f(0) )
    
    def DoublePowerLaw(self, Mh, logM1, N, beta, gamma):  # (arxiv.org/abs/1205.5807)  TODO 1
        M1 = 10**logM1
        return 2*N/( (Mh/M1)**(-beta) + (Mh/M1)**(gamma) )
    
    def SHMR(self, Ms, p={}):  # Interpolate to get halo masses from stellar mass
        Mhs = np.logspace(10, 18, 1000)
        Mss_from_Mhs = self.HSMR(Mhs, self.p0 | p)
        return np.interp(Ms, Mss_from_Mhs, Mhs)


class Xu2023(BASESHMR):  # SDSS Main DR7, CMASS & LOWZ DR12 (arxiv.org/abs/2211.02665)  TODO 1
    samples = ["Main_BP13", "LOWZ_BP13", "CMASS_BP13", "Main_DP", "LOWZ_DP", "CMASS_DP"]
    params = {
        'logM0': [11.338, 11.359, 11.509, 11.732, 11.579, 11.624],
        'alpha': [0.484, 0.623, 0.740, 0.299, 0.429, 0.466],
        'delta': [3.041, 3.248, 2.964, None, None, None],
        'beta': [1.632, 1.702, 2.094, 1.917, 2.215, 2.513],
        'logeps': [-1.545, -1.598, -1.565, None, None, None],
        'logk': [None, None, None, 10.303, 10.105, 10.133],
        'sigma': [0.237, 0.190, 0.190, 0.233, 0.201, 0.192]}

    def __init__(self, sample):
        self.checksample(sample)

    def HSMR(self, Mh, p={}):  # TODO 1
        p = self.p0 | p
        if self.sample in ["Main_BP13", "LOWZ_BP13", "CMASS_BP13"]:
            return self.Behroozi(Mh, logM1=p['logM0'], logeps=p['logeps'], alpha=-p['beta'], delta=p['delta'], gamma=p['alpha'])
        elif self.sample in ["Main_DP", "LOWZ_DP", "CMASS_DP"]:
            return self.DoublePowerLaw(Mh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])


class Gao2023(BASESHMR):  # DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)  TODO 1
    samples = ["ELG_Auto", "ELG_Cross", "Psat_Mh"]
    params = {
        'logM0': [11.56, 12.14, 12.07],
        'alpha': [0.43, 0.37, 0.37],
        'beta': [2.72, 2.27, 2.61],
        'logk': [10.11, 10.40, 10.36],
        'sigma': [0.18, 0.21, 0.21]}

    def __init__(self, sample):
        self.checksample(sample)

    def HSMR(self, Mh, p={}):
        p = self.p0 | p
        return self.DoublePowerLaw(Mh, logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])


class Kravstov2014(BASESHMR):  # SDSS DR8 (arxiv.org/abs/1401.7329)  TODO 1
    samples = ["M200c", "M200c_scatter", "M500c", "M500c_scatter", "M200m", "M200m_scatter", "Mvir", "Mvir_scatter"]
    params = {
        "logM1": [11.39, 11.35, 11.32, 11.28, 11.45, 11.41, 11.43, 11.39],
        "logeps": [-1.618, -1.642, -1.527, -1.556, -1.702, -1.720, -1.663, -1.685],
        "alpha": [1.795, 1.779, 1.856, 1.835, 1.736, 1.727, 1.750, 1.740],
        "delta": [4.345, 4.394, 4.376, 4.437, 4.273, 4.305, 4.290, 4.335],
        "gamma": [0.619, 0.547, 0.644, 0.567, 0.613, 0.544, 0.595, 0.531]}

    def __init__(self, sample):
        self.checksample(sample)

    def HSMR(self, Mh, p={}):
        p = self.p0 | p
        return self.Behroozi(Mh, logM1=p['logM1'], logeps=p['logeps'], alpha=-p['alpha'], delta=p['delta'], gamma=p['gamma'])
    

Classes = {
    "Xu2023": Xu2023,
    "Gao2023": Gao2023,
    "Kravstov2014": Kravstov2014,
}
def get_Class(class_name):
    print("Loading SHMR")
    try:
        return Classes[class_name]
    except KeyError:
        raise ValueError(f"Unknown class: {class_name}. Choose from {list(Classes.keys())}")