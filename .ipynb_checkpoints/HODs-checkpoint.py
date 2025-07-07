"""
Collections of Halo Occupancy Distribution model and parameterizations calibrated off different studies/samples. 
Classes should contain functions to calculate the average number of central and satellite galaxies (Nc/Ns) as a function of halo mass (in m200c) and the parameters values outlined in the papers, with other components of specific models (like density profile of satellite galaxies) added as needed.
Functions should also take an input dictionary to specify custom values for parameters, defaulting to the base values if not given. 
To more easily compare parameter values between models, they should be built off the Base Model functions.

# TODO: Check to see if I should be adding sattelite profiles for the rest of the HODs
# TODO: Check to make sure the halo masses are in m200c
"""

from Basics import *


class BaseHOD:  # Contains base functional forms and general utilities
    def checksample(self, sample):  # Set default parameterization from input sample, checking for validity
        if sample in self.samples:
            self.sample = sample
            self.p0 = {param: self.params[param][self.samples.index(self.sample)] for param in self.params.keys()}
        else:
            raise NameError(f"Sample {sample} doesn't exist, choose from available samples: {self.samples}")

    def N_cent(self, M, logM_min, sigma_logM):  # Expected number of centrals per halo (Zheng2005)
        return 0.5*(1+scipy.special.erf((np.log10(M)-logM_min)/sigma_logM))

    def N_sat(self, M, logM_0, logM_1, alpha):  # Expected number of satellites per halo (Zheng2005)
        return ((M-10**logM_0)/10**logM_1)**alpha

    def f_inc(self, M, alpha_inc, logM_inc):  # CMASS incompleteness function (More2015)
        return np.clip(1+alpha_inc*(np.log10(M)-logM_inc), 0, 1)
    
    def GNFW(self, x, gamma, alpha, beta):  # Satellite density profile
        return 1/(x**gamma * (1+x**alpha)**((beta-gamma)/alpha))


class Kou2023(BaseHOD):  # CMASS DR12 (arxiv.org/abs/2211.07502)
    samples = ["M*>10.8", "M*>11.1", "M*>11.25", "M*>11.4"]
    params = {
        "logM_min": [13.47, 13.58, 13.84, 14.20],  
        "sigma_logM": [0.76, 0.78, 0.86, 0.959],
        "logM_1": [14.119, 14.140, 14.171, 14.100],
        "beta_s": [4.38, 4.71, 5.31, 6.35],
        "1-b_h": [0.602, 0.623, 0.558, 0.550],
        "A": [0.981, 0.965, 0.956, 0.961],
        "alpha_inc": [0.51, 0.42, 0.39, 0.33],
        "logM_inc": [13.39, 13.42, 13.69, 13.96],
        "beta_m": [4.97, 5.91, 4.16, 10],
        }

    min_z, max_z, med_z = 0.47, 0.59, 0.53
    
    def __init__(self, sample):
        self.checksample(sample)

    def Nc(self, M, p={}):
        p = self.p0 | p
        return self.N_cent(M, logM_min=p['logM_min'], sigma_logM=p['sigma_logM']) * self.f_inc(M, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])

    def Ns(self, M):
        p = self.p0 | p
        return self.N_sat(M, logM_0=p['logM_min'], logM_1=p['logM_1'], alpha=1) * np.heaviside(M-10**p['logM_min'],1) * self.Nc(M, p)

    def uSat(self, x, p={}):
        p = self.p0 | p
        return self.GNFW(x, gamma=1, alpha=1, beta=p['beta_s'])


class Yuan2023(BaseHOD):  # DESI 1% LRGs/QSOs (arxiv.org/abs/2306.06314)
    samples = ["LRG 0.4<z<0.6", "LRG 0.6<z<0.8", "QSO 0.8<z<2.1"]
    params = {
        "logM_cut": [12.89, 12.78, 12.67],
        "logM_1": [14.08, 13.94, 15.00],
        "sigma": [0.27, 0.23, 0.58],
        "alpha": [1.20, 1.07, 1.09],
        "kappa": [0.65, 0.55, 0.74],
        "f_ic": [0.92, 0.89, 0.041],
        "f_sat": [0.089, 0.104, 0.05],
        "logM_h": [13.42, 13.26, 12.74],
        "b_lin": [1.94, 2.11, 2.56],
    }
    def __init__(self, sample):
        self.checksample(sample)
    
    def Nc(self, M, p={}):
        p = self.p0 | p
        return self.N_cent(M, logM_min=p['logM_cut'], sigma_logM=np.sqrt(2)*p['sigma']) * p['f_ic']
    
    def Ns(self, M, p={}):
        p = self.p0 | p
        if self.sample=="QSO 0.8<z<2.1":
            return self.N_sat(M, logM_0=np.log10(p['kappa'])+p['logM_cut'], logM_1 = p['logM_1'], alpha=p['alpha'])
        else:
            return self.N_sat(M, logM_0=np.log10(p['kappa'])+p['logM_cut'], logM_1 = p['logM_1'], alpha=p['alpha']) * self.Nc(M, p)


class Linke2022(BaseHOD):  # Millennium Simulation and KiDS+VIKING+GAMA (arxiv.org/abs/2204.02418)
    samples = ["MS red", "MS blue", "KV450 X GAMA red", "KV450 X GAMA blue"]
    params = {
        "alpha^a": [0.47, 0.10, 0.34, 0.13],
        "sigma^a": [0.55, 0.47, 0.52, 0.47],
        "M_th^a": [23.0, 1.19, 15, 1.4],
        "beta^a": [0.84, 0.73, 0.88, 0.55],
        "M^a": [5.8, 32, 3.6, 20],
        "f": [1.49, 0.88, 1.27, 0.83],
        "A": [5.31, 5.31, 1.62, 1.62],
        "epsilon": [0.69, 0.69, 0.99, 0.99],
    }

    def __init__(self, sample):
        self.checksample(sample)
        
    def Nc(self, M, p={}):
        p = self.p0 | p
        return self.N_cent(M, logM_min=p['M_th^a'], sigma_logM=p['sigma^a']) * p['alpha^a']
    
    def Ns(self, M, p={}):
        p = self.p0 | p
        return self.N_sat(M, logM_0=0, logM_1 = p['M^a'], alpha=p['beta^a']) * self.N_cent(M, logM_min=p['M_th^a'], sigma_logM=p['sigma^a'])
        

class Kusiak2022(BaseHOD):  # unWISE (arxiv.org/abs/2203.12583)
    samples = ["Blue", "Green", "Red"]
    params = {
        "sigma_logM": [0.73, 0.61, 0.75],
        "alpha_s": [1.38, 1.23, 1.18],
        "logM_min^HOD": [12.11, 12.39, 13.23],
        "logM_1": [13.00, 12.87, 12.20],
        "lambda": [1.11, 2.50, 1.30],
        "10^7A_SN": [-0.16, 1.35, 27.95],
    }

    def __init__(self, sample):
        self.checksample(sample)
    
    def Nc(self, M, p={}):
        p = self.p0 | p
        return self.N_cent(M, logM_min=p['logM_min^HOD'], sigma_logM=p['sigma_logM'])
    
    def Ns(self, M, p={}):
        p = self.p0 | p
        return self.N_sat(M, logM_0=0, logM_1 = p['M_1'], alpha=p['alpha_s']) * self.Nc(M, p)
    
    
class More2015(BaseHOD):  # CMASS DR11 (arxiv.org/abs/1407.1856)
    samples = ["[11.10, 12.00]", "[11.30, 12.0]", "[11.40, 12.0]"]
    params = {
        "logM_min": [13.13, 13.45, 13.68],
        "sigma^2": [0.22, 0.45, 0.79],
        "logM_1": [14.21, 14.51, 14.56],
        "alpha": [1.13, 1.14, 1.00],
        "kappa": [1.25, 0.85, 1.19],
        "M_stellar_11": [0, 0, 0],  # units of 10^11 h^(-2) M_solar
        "R_c": [0.98, 1.01, 1.02],
        "psi": [0.93, 0.93, 0.94],
        "p_off": [0.34, 0.37, 0.36],
        "R_off": [2.2, 2.3, 2.4],
        "alpha_inc": [0.44, 0.53, 0.57],
        "logM_inc": [13.57, 13.88, 14.08],
        "Omega_m": [0.310, 0.306, 0.304],
        "sigma_8": [0.785, 0.839, 0.813],
        "100*Omega_b*h^2": [2.228, 2.226, 2.222],
        "n_s": [0.964, 0.963, 0.961],
        "h": [0.703, 0.700, 0.695],
    }

    def __init__(self, sample):
        self.checksample(sample)

    def Nc(self, M, p={}):
        p = self.p0 | p
        return self.N_cent(M, logM_min=p['logM_min'], sigma_logM=p['sigma_logM']) * self.f_inc(M, alpha_inc=p['alpha_inc'], logMinc=p['logM_inc'])
    
    def Ns(self, M, p={}):
        p = self.p0 | p
        return self.N_sat(M, logM_0=np.log10(p['kappa'])+p['logM_min'], logM_1 = p['logM_1'], alpha=p['alpha']) * self.Nc(M, p)


Classes = {
    "Kou2023": Kou2023,
    "More2015": More2015,
    "Yuan2023": Yuan2023,
    "Linke2022": Linke2022,
    "Kusiak2022": Kusiak2022
}

def get_Class(class_name):
    print("Loading HOD")
    try:
        return Classes[class_name]
    except KeyError:
        raise ValueError(f"Unknown class: {class_name}. Choose from {list(Classes.keys())}")