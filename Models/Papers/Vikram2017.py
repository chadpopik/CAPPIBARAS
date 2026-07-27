"""
A Measurement of the Galaxy Group-Thermal Sunyaev-Zel'dovich Effect Cross-Correlation Function

ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
arxiv.org/pdf/1608.04160

NOTE: using colossus as halo model wasn't specified
"""


from config import *
import Models.HaloModels as HaloModels
from Models.Papers.colossus import colossus_model
from Models.Papers.PlotsTables import BasePlots2
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    """Throughout, we assume a ΛCDM model with ns = 1,σ8 = 0.8, Ωm = 0.27, ΩΛ = 0.73, Ωb = 0.044, and h = 0.7, broadly consistent with recent parameter determinations (Ade et al. 2013)."""
    ns = 1
    sigma8 = 0.8
    Om0 = 0.27
    Obl = 0.73
    Ob0 = 0.044
    h = 0.7
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        if not hasattr(self, 'hmod'):
            self.hmod = colossus_model(z=self.z, H0=self.h*100, Om0=self.Om0, Ob0=self.Ob0, sigma8=self.sigma8, ns=self.ns, flat=True)

        for key in ['H', 'rho_c']:
            if not hasattr(self, key):
                setattr(self, key, getattr(self, f"{key}_func")())

    def H_func(self):
        return self.hmod.H()

    def rho_c_func(self):
        return self.hmod.rhoc()


class HaloModel(Cosmology):        
    def __init__(self, inputdict={}, **inputvars):
        inputs = inputdict | inputvars
        M200c = inputs.get('M200c')
        self.hmod = colossus_model(
            z=inputs.get('z'), logM=None if M200c is None else np.log10(M200c), k=inputs.get('k'),
            H0=inputs.get('h', self.h)*100, Om0=inputs.get('Om0', self.Om0), Ob0=inputs.get('Ob0', self.Ob0),
            sigma8=inputs.get('sigma8', self.sigma8), ns=inputs.get('ns', self.ns), flat=True,
        )
        super().__init__(inputdict, **inputvars)

        # P_lin isn't precomputed here since it needs k, which isn't always given; call P_lin_func() explicitly when it's needed
        for key in ['c', 'R200c', 'dndM', 'b']:
            if not hasattr(self, key):
                setattr(self, key, getattr(self, f"{key}_func")())


    """with r200 denoting the radius at which the average matter density within the halo reaches 200 times the critical density. """
    def R200c_func(self):
        return (3*self.M200c*u.Msun/(4*np.pi*200*self.rho_c))**(1/3)

    """with the concentration parameter fit from Duffy et al. (2008)"""
    def c_func(self):
        return self.hmod.c("duffy08", "200c")

    """3.1 Plin(k) is the linear theory density power spectrum"""
    def P_lin_func(self):
        return self.hmod.Plin("eisenstein98")

    """3.1 dndM ′ denotes the mass function of the neighboring halos, while b(M ) and b(M ′) are the linear bias factors of the halos of mass M and M ′.We compute the mass function and bias factors using the formulae of Sheth & Tormen (2002) and Sheth, Mo & Tormen (2001), respectively
    NOTE: Sheth99/Sheth01 are calibrated on FoF halos, so MassDef='fof' here rather than '200c' (used for the concentration/profile); M200c is passed in as-is without converting mass definitions, following common practice in this style of two-halo term calculation"""
    def dndM_func(self):
        return self.hmod.dndlogm("sheth99", "fof")

    def b_func(self):
        return self.hmod.bh("sheth01", "fof")


    


class Profiles(HaloModel):
    def __init__(self, inputdict={}, **inputvars):
        super().__init__(inputdict, **inputvars)

        self.r = np.array(self.r, ndmin=1)[:, None, None]  # r axis (first), to broadcast against R200c's (z, M) shape

        for key in ['x', 'P200']:
            if not hasattr(self, key):
                setattr(self, key, getattr(self, f"{key}_func")())

        # try: self.p0 = Table1().getparams(ngbar=self.ngbar,Model=self.Model,numParams=self.numParams).to_dict()
        # except: self.p0 = {}

    """P200 is the thermal pressure assuming self-similarity: [Eq 6]"""
    def P200_func(self):
        return 200 * self.rho_c * self.Ob0/self.Om0 * c.G * self.M200c / self.R200c

    """expressing the results in terms of x = r/r200"""
    def x_func(self):
        return self.r / self.R200c
    
    """3.1 Battaglia et al. (2012) use the “generalized NFW profile” form to fit the pressure profiles in their simulations, expressing the results in terms of x = r/r200, with r200 denoting the radius at which the average matter density within the halo reaches 200 times the critical density The Battaglia et al. (2012) fit is: [Eq 5]"""
    def Pfit(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return self.P200 * p['P0'] * (self.x/p['xc'])**p['gamma'] * (1+(self.x/p['xc'])**p['alpha'])**(-p['beta'])
    
    

class twohalo(HaloModel):
    def __init__(self, inputdict={}, **inputvars):
        super().__init__(inputdict, **inputvars)
            
        # Define mass function and halo bias for masses of neighboring halos
        HaloModel_2h = HaloModel(inputdict | inputvars | {'M200c': self.M200c_2h})
        self.b_2h = HaloModel_2h.b
        self.dndM_2h = HaloModel_2h.dndM
        
        # Set FFT function which takes in a profile, and precompute P_lin (twohalo always needs both r and k, unlike a bare HaloModel)
        for key in ['FFT', 'P_lin']:
            if not hasattr(self, key):
                setattr(self, key, getattr(self, f"{key}_func")())
        
        # Precalculation for the 2halo integration
        self.twohalo_prefac = self.b_2h * self.P_lin
        self.twohalo_intfac = self.dndM_2h * self.b_2h
        
    def FFT_func(self):
        return HaloModels.mcfit_package(rs=self.r).FFT1D

    """Assuming linear-biasing, the two-halo term is: [Eq 8]"""
    def P_hp(self, Pe, **kwargs):
        u_P = self.FFT(Pe)
        return self.twohalo_prefac * np.trapezoid(self.twohalo_intfac*u_P, self.M200c_2h)
    
    

















class Fig3(BasePlots2):  # ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(1e-10, 6.7e-8), yscale='log'),
        dict(name='Fig3b', filename='Fig3b', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(1e-10, 7.7e-8), yscale='log'),
        dict(name='Fig3c', filename='Fig3c', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(1e-10, 1.9e-7), yscale='log'),
    ], [
        dict(name='Fig3d', filename='Fig3d', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(5e-10, 0.95e-6), yscale='log'),
        dict(name='Fig3e', filename='Fig3e', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(1e-9, 3.6e-6), yscale='log'),
        dict(name='Fig3f', filename='Fig3f', figsize=(16/3, 5),
             xlabel=r'$r$ [Mpc]', xlim=(0.01, 10), xscale='log',
             ylabel=r'$\xi^s_{y, g}(r)$', ylim=(1e-9, 1.9e-5), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)
