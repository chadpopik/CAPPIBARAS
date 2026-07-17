"""
On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum

ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
arxiv.org/pdf/1109.3711
"""

import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


class Cosmology():
    # 2. We adopt a flat tilted ΛCDM cosmology, with total matter density (in units of the critical) Ωm = ΩDM + Ωb = 0.25, baryon density Ωb = 0.043, cosmological constant ΩΛ = 0.75, a present day Hubble constant of H0 = 100h km s−1 Mpc−1, a scalar spectral index of the primordial power-spectrum ns= 0.96 and σ8 = 0.8.
    Om0 = 0.25
    Ob0 = 0.043
    Ol0 = 0.75
    H0 = 100
    ns = 0.96
    sigma8 = 0.8
    
    # 2. It is important to note that all masses and distances quoted in this work are given relative to
    h = 0.7
    
    # 3. where XH = 0.76 is the primordial hydrogen mass fraction
    XH = 0.76


class HaloModel():
    # 2. We adopt the standard working definition of cluster radii R∆as the radius at which the mean interior density equals ∆ times the critical density, ρcr(z) (e.g., for ∆ = 200 or 500).
    mdef = '200c'


# The normalized average pressure profiles and parametrized fits to these profiles from simulations with AGN feedback scaled by (r/R200)3, in mass bins (left panel) and redshift bins (right panel). Here we have independently fit each mass and redshift bin.
class Fig1(BasePlots2):
    subplots = [[
        dict(name='Fig1a', filename='Fig1a', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
        dict(name='Fig1b', filename='Fig1b', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Studies(BaseStudy):  # On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum, ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'ns':0.96, 'sigma8':0.8, 'h':0.7, 'XH':0.76,
        'mdef':'200c',  # Mass definition, S2p3/Eq11
    }


class Profiles(BaseProfile, Studies.Battaglia2011):  # Pressure Profile from GADGET-2 made hydro sims
    models = {}
    params = {        
        # best-fit GNFW pressure profile parameters, Table 1
        'P0_A0': 18.1, 'P0_alpham': 0.154, 'P0_alphaz': -0.758, # Amplitude 
        'xc_A0': 0.497, 'xc_alpham': -0.00865, 'xc_alphaz': 0.731,  # Core-scale
        'beta_pres_A0': 4.35, 'beta_pres_alpham': 0.0393, 'beta_pres_alphaz': 0.415, # Asymptotic fall off power law index
        # Fixed GNFW params, Section 4.1 paragraph 1
        'alpha_pres': 1, 
        'gamma_pres': -0.3,
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def MGNFW(self, x, P0, xc, gamma, alpha, beta):
        return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
    
    def PL(self, z, logM200c, A0, alpham, alphaz):
        return A0 * (10**logM200c/1e14)**alpham * (1+z)**alphaz
    
    def P200c(self, z, logM200c, units='cosmo'):  # Scaled pressure of 200c sphere, Section 4.1 paragraph 1
        P200c = c.G*(10**logM200c*u.Msun)*200*self.rhoc(z)/(2*self.r200c(z, logM200c))
        return self.Fb*P200c.to(self.units('pres', units))

    def Pressure(self, r, z, logM200c, units='cosmo'):  # B18 Eq. A1
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        P200c = self.P200c(z, logM200c, units)
        x = r*u.Mpc/self.r200c(z, logM200c)
        PGNFW = lambda p: self.MGNFW(x, gamma=p['gamma_pres'], alpha=p['alpha_pres'], 
                            P0=self.PL(z, logM200c, p['P0_A0'], p['P0_alpham'], p['P0_alphaz']), 
                            xc=self.PL(z, logM200c, p['xc_A0'], p['xc_alpham'], p['xc_alphaz']), 
                            beta=self.PL(z, logM200c, p['beta_pres_A0'], p['beta_pres_alpham'], p['beta_pres_alphaz']))
        return lambda p={}: P200c*PGNFW(self.p0 | p)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    def Fig1(self, width=16, height=6):
        return self.plot(filename=['Fig1a','Fig1b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$r/R_{200}$', ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$',
            xlim=(3e-2, 3), ylim=(8e-4, 2.35e-1), xscale='log', yscale='log')
        
    def Fig2(self, width=16, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$r/R_{200}$', ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$',
            xlim=(3e-2, 3), ylim=(8e-4, 2.35e-1), xscale='log', yscale='log')
