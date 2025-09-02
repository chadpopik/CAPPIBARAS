"""
Collections of Halo Occupancy Distribution model and parameterizations calibrated off different studies/samples.

Classes should contain functions to calculate the average number of central and satellite galaxies (Nc/Ns) as a function of halo mass (in m200c) and the parameters values outlined in the papers, with other components of specific models (like density profile of satellite galaxies) added as needed.

Functions should also take an input dictionary to specify custom values for parameters, defaulting to the base values if not given. 
To more easily compare parameter values between models, they should be built off the Base Model functions.
"""

import numpy as np
import scipy


class BaseHOD:
    # Check specification validity and assign parameters
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
        self.p0 = {param: self.params[param][self.samples.index(self.sample)] for param in self.params.keys()}

    # Expected number of centrals per halo (arxiv.org/abs/astro-ph/0408564)
    def Nc_Zheng2005(self, logM, logM_min, sigma_logM):  
        return 0.5*(1+scipy.special.erf((logM-logM_min)/sigma_logM))

    # Expected number of satellites per halo (arxiv.org/abs/astro-ph/0408564)
    def Ns_Zheng2005(self, logM, logM_0, logM_1, alpha):
        M, M_0, M_1 = 10**logM, 10**logM_0, 10**logM_1
        return ((M-M_0)/M_1)**alpha

    # CMASS incompleteness function (arxiv.org/abs/1407.1856)
    def f_inc_More2015(self, logM, alpha_inc, logM_inc):
        return np.clip(1+alpha_inc*(logM-logM_inc), 0, 1)

    # Default central density distribution (FFT of dirac delta)
    def uck(self):
        return 1

    # Default satellite distribution (FFT of NFW)
    def usk(self, rs, rs200c, FFTf):
        x = rs[:, None, None]/rs200c
        NFW = 1 / (x * (1+x)**2)
        return lambda p={}: FFTf(NFW)


class Kou2023(BaseHOD):  # CMASS DR12 (arxiv.org/abs/2211.07502)
    info = {'mdef': '200m', 'zmin': 0.47, 'zmax': 0.59, 'zmed': 0.53}
    
    samples = ["M*>10.8", "M*>11.1", "M*>11.25", "M*>11.4"]
    params = {
        "logM_min": [13.47, 13.58, 13.84, 14.20],  # minimum halo mass for a central galaxy/value at which halos contain 0.5 central galaxies on average
        "sigma_logM": [0.76, 0.78, 0.86, 0.959],
        "logM_1": [14.119, 14.140, 14.171, 14.100],  # controls the number of galaxies at high halo mass
        "beta_s": [4.38, 4.71, 5.31, 6.35],
        "1-b_h": [0.602, 0.623, 0.558, 0.550],
        "A": [0.981, 0.965, 0.956, 0.961],
        "alpha_inc": [0.51, 0.42, 0.39, 0.33],
        "logM_inc": [13.39, 13.42, 13.69, 13.96],
        "beta_m": [4.97, 5.91, 4.16, 10],
        }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        self.info['mstarmin'] = np.float32(self.sample[3:])

    def Nc(self, logM):
        func = lambda p: self.Nc_Zheng2005(logM, logM_min=p['logM_min'], sigma_logM=p['sigma_logM']) * self.f_inc_More2015(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        func = lambda p: self.Ns_Zheng2005(logM, logM_0=p['logM_min'], logM_1=p['logM_1'], alpha=1) * np.heaviside(logM-p['logM_min'],1) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)
    
    def uSat(self, rs, rs200c, FFTf):
        x = rs[:, None, None]/rs200c
        GNFW = lambda p: 1/(x * (1+x)**(p['beta_s']-1))
        func = lambda p: FFTf(GNFW(p))
        return lambda p={}: func(self.p0 | p)


class Yuan2023(BaseHOD):  # DESI 1% LRGs/QSOs (arxiv.org/abs/2306.06314)
    info = {'mdef': '200c', # M not clear, maybe same as zheng 2005/2007? or cmass?
            'mhalomin': 1.3e11,  # Msun/h
            }
    samples = ["LRG 0.4<z<0.6", "LRG 0.6<z<0.8", "QSO 0.8<z<2.1"]
    zmaxs= [0.6, 0.8, 2.1]
    zmins = [0.4, 0.6, 0.8]
    params = {
        "logM_cut": [12.89, 12.78, 12.67],  # Msun/h
        "logM_1": [14.08, 13.94, 15.00],  # Msun/h
        "sigma": [0.27, 0.23, 0.58],
        "alpha": [1.20, 1.07, 1.09],
        "kappa": [0.65, 0.55, 0.74],
        "f_ic": [0.92, 0.89, 0.041],
        "f_sat": [0.089, 0.104, 0.05],
        "logM_h_mean": [13.42, 13.26, 12.74],  # Msun/h
        "b_lin": [1.94, 2.11, 2.56],
    }

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
    
    def Nc(self, logM):
        func = lambda p: self.Nc_Zheng2005(logM, logM_min=p['logM_cut'], sigma_logM=np.sqrt(2)*p['sigma']) * p['f_ic']
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        if self.sample=="QSO 0.8<z<2.1":
            func = lambda p: self.Ns_Zheng2005(logM, logM_0=np.log10(p['kappa'])+p['logM_cut'], logM_1 = p['logM_1'], alpha=p['alpha'])
        else:
            func = lambda p: self.Ns_Zheng2005(logM, logM_0=np.log10(p['kappa'])+p['logM_cut'], logM_1 = p['logM_1'], alpha=p['alpha']) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)
        

class Kusiak2022(BaseHOD):  # unWISE (arxiv.org/abs/2203.12583)
    info = {'mdef': '200c', 
            'mhalomin': 7e8, 'mhalomax': 3.5e15,  # Msun/h
            'zmin_hmod': 0.005, 'zmax_hmod': 4,
            'zmin': 0, 'zmax': 2,
            'zmeans': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5}}
    
    samples = ["Blue", "Green", "Red"]
    params = {
        "sigma_logM": [0.73, 0.61, 0.75],
        "alpha_s": [1.38, 1.23, 1.18],
        "logM_min^HOD": [12.11, 12.39, 13.23],
        "logM_1": [13.00, 12.87, 12.20],
        "lambda": [1.11, 2.50, 1.30],
        "10^7A_SN": [-0.16, 1.35, 27.95],
    }
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        self.info['zmean'] = self.info['zmeans'][self.sample]
    
    def Nc(self, logM):
        func = lambda p: self.Nc_Zheng2005(logM, logM_min=p['logM_min^HOD'], sigma_logM=p['sigma_logM'])
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        func = lambda p: self.Ns_Zheng2005(logM, logM_0=0, logM_1 = p['logM_1'], alpha=p['alpha_s']) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)

    
    
class Linke2022(BaseHOD):  # Millennium Simulation and KiDS+VIKING+GAMA (arxiv.org/abs/2204.02418)
    info = {'mdef': '200m', 'zmax': 0.5,
            'mhalomin':10e11, 'mhalomax': 10e15,  # Msun/h^2, these cuts are just for sims
}
    # There are actually many further subsamples cut by stellar mass
    samples = ["MS red", "MS blue", "KV450 X GAMA red", "KV450 X GAMA blue"]
    params = {
        "alpha^a": [0.47, 0.10, 0.34, 0.13],
        "sigma^a": [0.55, 0.47, 0.52, 0.47],
        "M_th^a": [23.0, 1.19, 15, 1.4],  # 1e11 Msol
        "beta^a": [0.84, 0.73, 0.88, 0.55],
        "M^a": [5.8, 32, 3.6, 20],  # 1e13 Msol
        "f": [1.49, 0.88, 1.27, 0.83],
        "A": [5.31, 5.31, 1.62, 1.62],
        "epsilon": [0.69, 0.69, 0.99, 0.99],
    }
        
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])
        
    def Nc(self, logM):
        func = lambda p: self.Nc_Zheng2005(logM, logM_min=np.log10(p['M_th^a']*1e11), sigma_logM=p['sigma^a']) * p['alpha^a']
        return lambda p={}: func(self.p0 | p)
    
    def Ns(self, logM):
        func = lambda p: self.Ns_Zheng2005(logM, logM_0=0, logM_1 = np.log10(p['M^a']*1e13), alpha=p['beta^a']) * self.Nc_Zheng2005(logM, logM_min=np.log10(p['M_th^a']*1e11), sigma_logM=p['sigma^a'])
        return lambda p={}: func(self.p0 | p)


class More2015(BaseHOD):  # CMASS DR11 (arxiv.org/abs/1407.1856)
    info = {'mdef': '200m',  # M200b, 200 times overdense wrt background matter density
            }
    
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
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample'])

    def Nc(self, logM):
        func = lambda p: self.Nc_Zheng2005(logM, logM_min=p['logM_min'], sigma_logM=p['sigma^2']**0.5) * self.f_inc_More2015(logM, alpha_inc=p['alpha_inc'], logM_inc=p['logM_inc'])
        return lambda p={}: func(self.p0 | p)

    def Ns(self, logM):
        func = lambda p:  self.Ns_Zheng2005(logM, logM_0=np.log10(p['kappa'])+p['logM_min'], logM_1 = p['logM_1'], alpha=p['alpha']) * self.Nc(logM)(p)
        return lambda p={}: func(self.p0 | p)
