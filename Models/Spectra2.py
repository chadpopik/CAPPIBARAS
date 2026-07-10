import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import RegularGridInterpolator

import Models.FFTs as FFTs
import Models.Studies as Studies
import Models.HODs as HODs
import Models.TargetData as TargetData
import Models.MapData as MapData
    
    
    
class BaseSpectra:
    def __init__(self):
        pass


class Popik2026(BaseSpectra, Studies.Popik2026):  
    def __init__(self, rs, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

        # Define FFTs and get ks from the FFT functions
        self.fft = FFTs.mcfit_package(rs)
        self.ks = self.fft.ks/u.Mpc
        self.FFT3D, self.IFFT1D = self.fft.FFT3D, self.fft.IFFT1D
        
    def HODweighting(self, HOD, hmod, targets, **kwargs):
        logM = targets.logMh
        zs = targets.z

        Nc, Ns = HOD.Ncen(logM), HOD.Nsat(logM)
        nc, ns = HOD.ncen(self.ks, zs, logM), HOD.nsat(self.ks, zs, logM)

        dndlogm = hmod.dndlogm(zs[:, None], logM)  # calculate halo mass function
        # dndlogm[:, (targets.logM<12) | (targets.logM>16)] = 0
        infac = dndlogm*targets.dNdz[:, None]  # combined mass/z distributions

        ngal = lambda p: np.trapz((Nc(p)+Ns(p))*dndlogm, logM)  # total galaxy number
        Hg = lambda p: (Nc(p)*nc(p) + Ns(p)*ns(p))/ngal(p)[None, :, None]  # HOD cross-spectra function
        # Hg = lambda p: (Nc(p)[None, None, :] + Ns(p)[None, None, :])/ngal(p)[:, None]  # HOD cross-spectra function
        Hg_norm = lambda p: Hg(p)/np.trapz(np.trapz(Hg(p)*infac, logM), zs)[:, None, None]  # normalized galaxy distribution
        intfac0 = Hg_norm({})*infac  # combine default HOD galaxy dist into integrand factor
        intfac = lambda p: Hg_norm(p)*infac if p!={} else intfac0  # recalculate integrand factor if HOD galaxy dist is being fit
    
        aveprof = lambda prof, p: np.trapz(np.trapz(self.FFT3D(prof)*intfac(p), logM), zs) # take mass/redshift average

        return lambda prof, p={}: self.IFFT1D(aveprof(prof, p))*prof.unit


    
    def HODweightingM(self, HOD, hmod, targets, **kwargs):
        logM = targets.logMh
        zs = targets.z

        Nc, Ns = HOD.Ncen(logM), HOD.Nsat(logM)
        dndlogm = hmod.dndlogm(zs[:, None], logM)  # calculate halo mass function
        # dndlogm[:, (targets.logM<12) | (targets.logM>16)] = 0
        infac = dndlogm*targets.dNdz[:, None]  # combined mass/z distributions

        ngal = lambda p: np.trapz((Nc(p)+Ns(p))*dndlogm, logM)  # total galaxy number
        Hg = lambda p: (Nc(p) + Ns(p))/ngal(p)[:, None]  # HOD cross-spectra function
        # Hg = lambda p: (Nc(p)[None, None, :] + Ns(p)[None, None, :])/ngal(p)[:, None]  # HOD cross-spectra function
        Hg_norm = lambda p: Hg(p)/np.trapz(np.trapz(Hg(p)*infac, logM), zs)  # normalized galaxy distribution
        intfac0 = Hg_norm({})*infac  # combine default HOD galaxy dist into integrand factor
        intfac = lambda p: Hg_norm(p)*infac if p!={} else intfac0  # recalculate integrand factor if HOD galaxy dist is being fit
    
        return lambda prof, p={}: np.trapz(np.trapz(prof*intfac(p), logM), zs) # take mass/redshift average
    
    