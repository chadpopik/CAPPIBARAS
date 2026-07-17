"""
The DESI One-Percent Survey: Constructing Galaxy-Halo Connections for ELGs and LRGs Using Auto and Cross Correlations

ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
arxiv.org/pdf/2306.06317
"""


import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


class Cosmology():
    # fixed cosmo info
    h = 0.71
    Om0 = 0.268
    Ol0 = 0.732


class HaloModel():
    MassDef = 'vir'  # Current Virial Mass


class SHMR():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        try: self.p0 = Table3().getparams(model=self.model).to_dict()
        except: self.p0 = {}

    # Section 4. Double power-law form fit to the SHMR for all three Psat models (Table 3).
    def log10Mstar(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        logMh = self.logMh - np.log10(Cosmology.h)  # Mh/h -> Mh
        Mh, M0, k = 10**logMh, 10**p['log10M0'], 10**p['log10k']

        Mstar = 2*k / ((Mh/M0)**(-p['beta']) + (Mh/M0)**(-p['alpha']))
        return np.log10(Mstar)


# TABLE 3. Best-fit parameters of the SHMRs for different Psat models.
class Table3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class Fig7a(BasePlots2):
    subplots = [[
        dict(name='Fig7a', filename='Fig7a', figsize=(8, 6),
             xlabel=r'$\log(M_h) \ [M_\odot /h]$', xlim=(10, 15), xscale='linear',
             ylabel=r'$\log(M_*) \ [M_\odot]$', ylim=(7.5, 12), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(9, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(5, 3.5),
             xlabel=r'$\log M_* \ [M_\odot]$', xlim=(7, 12.5), xscale='linear',
             ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$', ylim=(5e-7, 2e-3), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
    subs = {}
    info = {        
        # fixed cosmo info
        'h':0.71, 'Om0':0.268, 'Ol0':0.732,
        'mdef':'vir',  # Current Virial Mass
        'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
    }
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)


class TargetData(BaseTargetData, Studies.Gao2023):  # DESI 1% LRGs and ELGs (Gao+ 2023, arxiv.org/abs/2306.06317)
    path = f"{datapath}/Gao2023"
    subs = {'sample':['LRG', 'ELG']}  # Galaxy Sample

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])
        
        self.make_zdists

    def make_zMsdists(self, dz=None, zMin=None, zMax=None, dlogMs=None, logMsMin=None, logMsMax=None):
        if self.sample=='LRG': zbins = np.arange(0.4, 1.2, 0.1)
        elif self.sample=='ELG': zbins = np.arange(0.6, 1.6, 0.1)
        self.z_df = (zbins[1:]+zbins[:-1])/2
        self.dz_df = self.z_df[1]-self.z_df[0]
        
        # Read the plot data from the files
        self.logMs_df = pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.n_logMs_z_h3 = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in zbins[:-1]])  # [(Mpc/h)^-3 dex^-1]
        
        self.dndzdlogMs_df = self.n_logMs_z_h3 *self.h**3 /u.Mpc**3 /self.dz_df/u.dex
        
        hmod = HaloModels.astropy_model(**Studies.Gao2023.info)
        Vcoms = (hmod.Vcom(self.z_df+self.dz_df/2)-hmod.Vcom(self.z_df-self.dz_df/2)) *(self.area/(4*np.pi*u.sr).to(u.deg**2))  # Calculate non-comoving shell for every z
        self.dNdzdlogMs_df = self.dndzdlogMs_df * Vcoms[:, None]
        
        dNinterp = RegularGridInterpolator((self.z_df, self.logMs_df), self.dNdzdlogMs_df,bounds_error=False, fill_value=0)

        zmin = zMin if zMin is not None else self.z_df.min()
        zmax = zMax if zMax is not None else self.z_df.max()
        self.dz = dz if dz is not None else self.z_df[1]-self.z_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)

        logMsMin = logMsMin if logMsMin is not None else self.logMs_df.min()
        logMsMax = logMsMax if logMsMax is not None else self.logMs_df.max()
        self.dlogMs = dlogMs if dlogMs is not None else self.logMs_df[1]-self.logMs_df[0]
        self.logMs = np.arange(logMsMin, logMsMax+self.dlogMs, self.dlogMs)

        zgrid, logMsgrid = np.meshgrid(self.z, self.logMs, indexing='ij')
        self.dNdzdlogMs = dNinterp(np.column_stack([zgrid.ravel(), logMsgrid.ravel()])).reshape(len(self.z), len(self.logMs)) / u.dex
        
        self.dNdogMs_z = self.dNdzdlogMs *self.dz
        self.N_z = np.trapz(self.dNdogMs_z, self.logMs)
        self.n_z = self.N_z / self.area
        self.dNdz = self.N_z / self.dz
        self.dndz = self.dNdz / self.area
        
        self.N_z_logMs = self.dNdogMs_z *self.dlogMs*u.dex


class ParamsTable(ParamTable):  # best-fit SHMR parameters, Table 3
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)


class SHMRs(BaseSHMR, Studies.Gao2023):  # DESI 1% (arxiv.org/abs/2306.06317)
    models = {'model':["Auto", "Cross", "Psat"],}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = ParamsTable().getparams(model=self.model).to_dict()

    def SHMR(self, logMh):
        self.require(['model'])
        func = lambda p: self.DoublePowerLaw(logMh-np.log10(self.h), logM1=p['logM0'], N=10**p['logk'], beta=p['beta'], gamma=-p['alpha'])
        return lambda p={}: func(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
    def Fig7a(self, width=8, height=6):
        return self.plot(filename='Fig7a', width=width, height=height,
            xlabel=r'$\log(M_h) \ [M_\odot /h]$', ylabel=r'$\log(M_*) \ [M_\odot]$',
            xlim=(10, 15), ylim=(7.5, 12), xscale='linear', yscale='linear')

    def Fig2(self, width=10, height=3.5):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$\log M_* \ [M_\odot]$', ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$',
            xlim=[(9, 12.5), (7, 12.5)], ylim=(5e-7, 2e-3), xscale='linear', yscale='log')
