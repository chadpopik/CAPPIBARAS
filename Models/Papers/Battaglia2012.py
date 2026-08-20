"""
On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum

ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
arxiv.org/pdf/1109.3711
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Battaglia2012")


class Cosmology():
    p0 = {
    # 2. We adopt a flat tilted ΛCDM cosmology, with total matter density (in units of the critical) Ωm = ΩDM + Ωb = 0.25, baryon density Ωb = 0.043, cosmological constant ΩΛ = 0.75, a present day Hubble constant of H0 = 100h km s−1 Mpc−1, a scalar spectral index of the primordial power-spectrum ns= 0.96 and σ8 = 0.8.
    'Om0' : 0.25,
    'Ob0' : 0.043,
    'Ol0' : 0.75,
    'H0' : 100,
    'ns' : 0.96,
    'sigma8' : 0.8,
    # 2. It is important to note that all masses and distances quoted in this work are given relative to
    'h' : 0.7,
    # 3. where XH = 0.76 is the primordial hydrogen mass fraction
    'XH' : 0.76,
    }
    
    p0['H0'] = 100*u.km/u.s/u.Mpc * p0['h']
        
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        self.rhoc = self.rhoc_func()
        
    # critical density in cgs
    def rhoc_func(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return (3*p['H0']**2/(8*np.pi*c.G)*(p['Om0']*(1+self.z)**3 + p['Ol0'])).to(u.g/u.cm**3)


class HaloModel(Cosmology):
    # 2. We adopt the standard working definition of cluster radii R∆as the radius at which the mean interior density equals ∆ times the critical density, ρcr(z) (e.g., for ∆ = 200 or 500).
    mdef = '200c'
    
    def __init__(self, inputdict={}, **inputvars):
        super().__init__(inputdict, **inputvars)
        self.r200c = self.r200c_func()
        
    # radius of a sphere with density 200 times the critical density of the universe. Input mass in solar masses. Output radius in cm.
    def r200c_func(self):
        return ((3*10**self.logM200c*u.Msun/(4*np.pi*200*self.rhoc))**(1/3)).to(u.Mpc)



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


# TABLE 1. Mass and redshift fit parameters for the pressure profile amplitude (P0), core-scale (xc), and outer slope (beta), from Eqns. (10) and (11).
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        self.df = pd.read_csv(filename, index_col=0)

    def getparams(self):
        namemap = {'P0': 'P0', 'xc': 'xc', 'beta': 'beta_pres'}
        p0 = {}
        for param, code in namemap.items():
            row = self.df.loc[param]
            p0[f'{code}_A0'] = row['A']
            p0[f'{code}_alpham'] = row['alpha_m']
            p0[f'{code}_alphaz'] = row['alpha_z']
        return p0


class HaloProfiles_new(HaloModel):  # Pressure Profile from GADGET-2 made hydro sims
    def setdim(self, r, z, logM200c):  # Set proper dimensions of r, z, logM200c: [nr,1,1], [nz,1], [nM]
        r = r if np.array(r, ndmin=1).ndim==3 else np.array(r, ndmin=1)[:, None, None]
        z = z if np.array(z, ndmin=1).ndim==2 else np.array(z, ndmin=1)[:, None]
        logM200c = logM200c if np.array(logM200c, ndmin=1).ndim==1 else np.array(logM200c, ndmin=1)
        return r, z, logM200c

    def __init__(self, logM200c, z, r, units='cosmo', inputdict={}, **inputvars):
        self.r, self.z, self.logM200c = self.setdim(r, z, logM200c)
        super().__init__(inputdict, **inputvars)  # applies any cosmological overrides (H0, Om0, etc.), then computes self.rhoc, self.r200c
        self.Fb = self.Ob0/self.Om0

        self.p0 = {'alpha_pres': 1, 'gamma_pres': -0.3}  # Fixed GNFW params, Section 4.1 paragraph 1
        try: self.p0 |= Table1().getparams()
        except: pass

        P200c = c.G*(10**self.logM200c*u.Msun)*200*self.rhoc/(2*self.r200c)
        self.P200c = self.Fb*P200c.to(self.units('pres', units))  # Scaled pressure of 200c sphere, Section 4.1 paragraph 1
        self.x = self.r*u.Mpc/self.r200c

    def units(self, prof, units):  # handles units of pth for cosmo and cgs
        if prof=='pres':
            if units=='cosmo': return u.Msun/u.Mpc/u.s**2
            elif units=='cgs': return u.g/u.cm/u.s**2
            elif units=='kpc': return u.Msun/u.kpc/u.s**2

    def MGNFW(self, x, P0, xc, gamma, alpha, beta):  # Eq 10
        return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)

    def PL(self, A0, alpham, alphaz):  # Eq 11
        return A0 * (10**self.logM200c/1e14)**alpham * (1+self.z)**alphaz

    def Pressure(self, pdict={}, **kwargs):  # B18 Eq. A1
        p = self.p0 | pdict | kwargs
        return self.P200c * self.MGNFW(self.x, gamma=p['gamma_pres'], alpha=p['alpha_pres'],
                            P0=self.PL(p['P0_A0'], p['P0_alpham'], p['P0_alphaz']),
                            xc=self.PL(p['xc_A0'], p['xc_alpham'], p['xc_alphaz']),
                            beta=self.PL(p['beta_pres_A0'], p['beta_pres_alpham'], p['beta_pres_alphaz']))



"""Everything below this line is old implementation which I'm trying to phase out"""

from Models.Studies import BaseStudy
class Study(BaseStudy):  # On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum, ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'ns':0.96, 'sigma8':0.8, 'h':0.7, 'XH':0.76,
        'mdef':'200c',  # Mass definition, S2p3/Eq11
    }
    
from CAPPIBARAS.Models.OldModules.Profiles import BaseProfile
class HaloProfiles(BaseProfile, Study):  # Pressure Profile from GADGET-2 made hydro sims
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