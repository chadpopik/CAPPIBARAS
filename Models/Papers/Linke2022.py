"""
KiDS+VIKING+GAMA: Halo occupation distributions and correlations of satellite numbers with a new halo model of the galaxy-matter bispectrum for galaxy-galaxy-galaxy lensing

ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
arxiv.org/pdf/2204.02418
"""


from config import *

from scipy.special import erf

from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Linke2022")


class Cosmology():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


class HOD_new():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


    def Ncen(self, pdict={}, **kwargs): # Eq 36
        p = self.p0 | pdict | kwargs
        return (p['alpha_a']/2) * (1+erf((self.logM-p['logMtha'])/p['sigmaa']))


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(8, 6),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'$\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8a', filename='Fig8a', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
        dict(name='Fig8b', filename='Fig8b', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # best fit params + Cosmo params, per sample (color-independent)
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)



class HODParamsTable(ParamTable):  # Best-fit parameters, Table 3
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        super().__init__(filename)




"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    subs = {'sample':['MS', 'KVG'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    info = {
        # best fit params, unclear what for:
        'A': {'MS': {'r': 5.31, 'b': 5.31}, 'KVG': {'r': 1.62, 'b': 1.62}},
        'epsilon': {'MS': {'r': 0.69, 'b': 0.69}, 'KVG': {'r': 0.99, 'b': 0.99}},

        # Cosmo params 1.p8
        'Om0':{'MS': 0.25, 'KVG': 0.315}, 'Ob0':{'MS': 0.045, 'KVG': 0.049}, 'H0':{'MS': 73, 'KVG': 67.4}, 'sigma8':{'MS': 0.9, 'KVG': 0.811},
        'mdef': '200m', 'zMax': 0.5,
        # Msun/h^2, these cuts are just for sims
        'MhMin':{'MS': 10e11}, 'MhMax': {'MS':10e15},
    }
    info['H0'] = cycle(info['H0'], lambda H: H *u.km/u.s/u.Mpc)
    info['MhMin'] = cycle(info['MhMin'], lambda M: M *u.Msun)
    info['MhMax'] = cycle(info['MhMax'], lambda M: M *u.Msun)
    
    
    
    
from Models.HODs import BaseHOD
from Models.Papers import Zheng2005
from Models import HaloModels
class HOD(BaseHOD, Study):  # arxiv.org/abs/2204.02418
    models = {'sample':['MS', 'KVG', 'fid'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    params ={
        # Best-fit parameters, Table 3
        'alpha_a': {'MS': {'r': 0.47, 'b': 0.1}, 'KVG': {'r': 0.34, 'b': 0.13}},  # for the maximum of〈N^a|m〉, gives the fraction of massive halos (m >> M^a_th) with a central galaxy from population a
        'sigma_a': {'MS': {'r': 0.55, 'b': 0.47}, 'KVG': {'r': 0.52, 'b': 0.47}},  # determines the transition of 〈N^a_cen | m〉from 0 to α^a
        'M_th_a': {'MS': {'r': 23.0, 'b': 1.19}, 'KVG': {'r': 15, 'b': 1.4}},  # halo mass below which we do not expect halos to contain galaxies, 1e11 Msol
        'beta_a': {'MS': {'r': 0.84, 'b': 0.73}, 'KVG': {'r': 0.88, 'b': 0.55}},
        'M_a': {'MS': {'r': 5.8, 'b': 32}, 'KVG': {'r': 3.6, 'b': 20}},  # 1e13 Msol
        'f_a': {'MS': {'r': 1.49, 'b': 0.88}, 'KVG': {'r': 1.27, 'b': 0.83}},  # factor in difference of concentration used in NFW from halo matter profile to average number density of satellite galaxies
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Ncen(self, logM):  # Eq 36
        func = lambda p: Zheng2005.HOD().Nc(logM, logMmin=np.log10(p['M_th_a']*1e11), sigmalogM=p['sigma_a']) * p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005.HOD().Ns(10**logM/self.h/self.h, M0=0, M1 = p['M_a']*1e13, alpha=p['beta_a']) * self.Ncen(logM)(p)/p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def ns_r(self, rs, zs, logMs):
        rs, zs, logMs = self.setdim(rs, zs, logMs)
        rdels, cdels = self.r200m(zs, logMs), self.c200m(zs, logMs)
        xs = rs*u.Mpc/rdels
        NFW_func = 1/(xs*(1/cdels+xs)**2)
        A_NFW = (1+cdels)/(np.log(1+cdels)*(1+cdels)-cdels)
        rhom = self.rhoc(zs)*self.Omega_m
        val = 200*rhom/3 *NFW_func * A_NFW/(10**logMs*u.Msun)
        return lambda **kwargs: val

    def ns(self, ks, zs, logMs):
        self.require(['rhoc', 'c200m', 'r200m'])
        fft = HaloModels.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)()
        val = fft.FFT3D(NFW)
        return lambda **kwargs: val