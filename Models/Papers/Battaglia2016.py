"""
The tau of galaxy clusters

ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
arxiv.org/pdf/1607.02442
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Battaglia2016")


class Cosmology():
    
    
    #     info = {
    #     # sim's cosmo params, B15.3.P2
    #     'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
    #     'MassDef':'200c',  # Mass definition, B15.T2
    #     'MassFunc': 'Tinker08',
    # }
        
    pass


class HaloModel():
    pass


class Fig5(BasePlots2):
    subplots = [[
        dict(name='Fig5a', filename='Fig5a', figsize=(7, 6),
             xlabel=r'$x=r/R_{200}$', xlim=(7e-2, 4), xscale='log',
             ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$', ylim=(1e1, 1e2), yscale='log'),
        dict(name='Fig5b', filename='Fig5b', figsize=(7, 6),
             xlabel=r'$x=r/R_{200}$', xlim=(7e-2, 4), xscale='log',
             ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$', ylim=(1e1, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)



class ParamsTable(ParamTable):  # best-fit GNFW parameters, Table 2
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)





"""Everything below this line is old implementation which I'm trying to phase out"""

from Models.Studies import BaseStudy
class Study(BaseStudy):  # The tau of galaxy clusters, ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
    subs={}
    info = {
        # sim's cosmo params, B15.3.P2
        'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
        'MassDef':'200c',  # Mass definition, B15.T2
        'MassFunc': 'Tinker08',
    }
    
    
    
from CAPPIBARAS.Models.OldModules.Profiles import BaseProfile
class HaloProfiles(BaseProfile, Study):  # Density Profile from GADGET-2 hydro sims
    models = {'model':['AGN', 'SH'],}  # AGN feedback vs shock heating sub-grid physics models
    params = {
        # best-fit GNFW parameters, Table 2
        'rho0_A0': {'AGN':4e3, 'SH':1.9e4},   # density amplitude
        'rho0_alpham': {'AGN':0.29, 'SH':0.09},
        'rho0_alphaz': {'AGN':-0.66, 'SH':-0.95},
        'alpha_A0': {'AGN':0.88, 'SH':0.70},   # density intermediate slope
        'alpha_alpham': {'AGN':-0.03, 'SH':-0.017},
        'alpha_alphaz': {'AGN':0.19, 'SH':0.27},
        'beta_A0': {'AGN':3.83, 'SH':4.43}, # density asymptotic fall off power law index
        'beta_alpham': {'AGN': 0.04, 'SH':0.005},
        'beta_alphaz': {'AGN':-0.025, 'SH':0.037},
        # fixed GNFW parameters, fixed GNFW params, B15.A.P2
        'xc': 0.5, 
        'gamma': -0.2,
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def GNFW(self, x, rho0, xc, gamma, alpha, beta):  # Eq A1
        # NOTE: sign in exponent is different from paper, which has a typo, should be beta+gamma
        return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta+gamma)/alpha)

    def pc(self, z, units='cosmo'):  # right before eq A1
        return self.Fb*self.rhoc(z).to(self.units('dens', units))  # prefactor and units
    
    def PL(self, z, logM200c, A0, alpham, alphaz):  # Eq A2
        return A0 * (10**logM200c/1e14)**alpham * (1+z)**alphaz

    def Density(self, r, z, logM200c, units='cosmo'):
        self.require(['model'])
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        pc = self.pc(z, units)
        x = r*u.Mpc/self.r200c(z, logM200c)
        pGNFW = lambda p: self.GNFW(x, gamma=p['gamma'], xc=p['xc'],
                                alpha=self.PL(z, logM200c, p['alpha_A0'], p['alpha_alpham'], p['alpha_alphaz']),
                                rho0=self.PL(z, logM200c, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
                                beta=self.PL(z, logM200c, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: pc*pGNFW(self.p0 | p)