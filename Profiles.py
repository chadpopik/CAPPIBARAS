"""
Collections of radial profiles (pressure/density so far) either from functional forms, constructed from an emulator, or loaded in from a data file, both one-halo and two-halo, for various studies and their fixed/inferred parameters.
Functions should return lambda functions which take in parameter dictionaries, so that parts of the calculation unreliant on sampled parameters don't have to be redone. These lambda functions should return 3D arrays over radius, halo mass (in m200c), and redshift, even if they don't use those mass/redshift arrays for a calculation, to ensure same dimensionality between one-halo and two-halo profiles.


# TODO 1: Add a two-halo term that's constructed from theory?
"""

from Basics import *

class BaseGNFW:
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):  # Check if the model is in the list of models
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
        self.p0 = {param: self.params[param][self.models.index(self.model)] for param in self.params.keys()}
    
    def PLmz(self, m200c, z, A0, alpham, alphaz):
        return A0 * (m200c/1.e14)**alpham * (1.+z)**alphaz
    
    def rho_over_rhodel(self, x, rho0, xc, gamma, alpha, beta):
        return rho0* (x/xc)**gamma * (1.+(x/xc)**alpha)**(-(beta-gamma)/alpha)
    
    def Pth_over_Pdel(self, x, P0, xc, gamma, alpha, beta):
        return P0* (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)


class Amodeo2021(BaseGNFW):  # BOSS DR10 cross-correlated with ACT DR5 (arxiv.org/abs/2009.05558)
    mdef = "M200c"
    meanmass, medz = 3.3*10**13, 0.55
    models = ['GNFW']
    params = {'logrho0': [2.6], 
              'xc_k': [0.6],
              'beta_k': [2.6],
              'A2h_k': [1.1],
              'P0': [2.0],
              'alpha_t': [0.8],
              'beta_t': [2.6],
              'A2h_t': [0.7]
    }
    twohalofile = '/global/homes/c/cpopik/Capybara/Data/twohalo_cmass_average.txt'

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])
        
        self.rs2hfile, self.rho2hfile, self.pth2hfile = np.genfromtxt(self.twohalofile, unpack=True)

    def Pth1h(self, r, m200c, z, rhocrit_func, r200c_func, cosmopars):
        # Components not reliant on r are calculated beforehand to avoid being done every time
        r, m200c = r[:, None, None], m200c[:, None]
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value
        p200c = G_cosmo*m200c*200*rhocrit_func(z)/(2*r200c_func(m200c, z))
        x = r/r200c_func(m200c, z)[None, ...]
        unitconv = (u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)
        frac_b = cosmopars['Omega_b']/cosmopars['Omega_m']* 2.0*(cosmopars['XH']+1.0)/(5.0*cosmopars['XH']+3.0)
        factorfront = unitconv*frac_b*p200c
        
        func = lambda p: self.Pth_over_Pdel(x, gamma=-0.3, alpha=p['alpha_t'], P0 = p['P0'], xc=self.PLmz(m200c, z, 0.497, -0.00865, 0.731), beta=p['beta_t'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def rho1h(self, r, m200c, z, rhocrit_func, r200c_func, cosmopars):
        r, m200c = r[:, None, None], m200c[:, None]
        x = r/r200c_func(m200c, z)[None, ...]
        frac_b = cosmopars['Omega_b']/cosmopars['Omega_m']
        rhocs = rhocrit_func(z)
        factorfront = frac_b*rhocs*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)
        func = lambda p: self.rho_over_rhodel(x, gamma=-0.2, alpha=1, rho0=10**p['logrho0'], xc=p['xc_k'], beta=p['beta_k'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def Pth2h(self, r, m, z):  # TODO 1
        Pth2h = np.interp(r, self.rs2hfile, self.pth2hfile)[:, None, None]*np.ones((r.size, m.size, z.size))
        func = lambda p: p['A2h_t']*Pth2h
        return lambda p={}: func(self.p0 | p)
    
    def rho2h(self, r, m, z):  # TODO 1
        rho2h = np.interp(r, self.rs2hfile, self.rho2hfile)[:, None, None]*np.ones((r.size, m.size, z.size))
        func = lambda p: p['A2h_k']*rho2h
        return lambda p={}: func(self.p0 | p)


class Battaglia2015(BaseGNFW):  # SPH sims made from GADGET-2 (arxiv.org/abs/1607.02442)
    mdef = "M200c"
    models = ['AGN', 'SH']
    params = {'rho0_A0': [4*1e3, 1.9*1e4], 
                'rho0_alpham': [0.29, 0.09], 
                'rho0_alphaz': [-0.66, -0.95],
                'alpha_A0': [0.88, 0.70], 
                'alpha_alpham': [-0.03, -0.017], 
                'alpha_alphaz': [0.19, 0.27],
                'beta_A0': [3.83, 4.43], 
                'beta_alpham': [0.04, 0.005], 
                'beta_alphaz': [-0.025, 0.037]}
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])
        
    def rho1h(self, r, m200c, z, rhocrit_func, r200c_func, cosmopars):
        r, m200c = r[:, None, None], m200c[:, None]
        x = r/r200c_func(m200c, z)[None, ...]
        frac_b = cosmopars['Omega_b']/cosmopars['Omega_m']
        rhocs = rhocrit_func(z)
        factorfront = frac_b*rhocs*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)
        func = lambda p: self.rho_over_rhodel(x, gamma=-0.2, 
                                            alpha=self.PLmz(m200c, z, p['alpha_A0'], p['beta_alpham'], p['beta_alphaz']),
                                            rho0=self.PLmz(m200c, z, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
                                            xc=0.5,
                                            beta=self.PLmz(m200c, z, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)


class Battaglia2012(BaseGNFW):  # SPH sims made from GADGET-2 (arxiv.org/abs/1109.3711)
    mdef = "M200c"
    models = ['B12']
    params = {'P0_A0': [18.1], 
                'P0_alpham': [0.154], 
                'P0_alphaz': [-0.758],
                'xc_A0': [0.497], 
                'xc_alpham': [-0.00865], 
                'xc_alphaz': [0.731],
                'beta_A0': [4.35], 
                'beta_alpham': [0.0393], 
                'beta_alphaz': [0.415]}
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])

    def Pth1h(self, r, m200c, z, rhocrit_func, r200c_func, cosmopars):
        # Components not reliant on r are calculated beforehand to avoid being done every time
        r, m200c = r[:, None, None], m200c[:, None]
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value
        p200c = G_cosmo*m200c*200*rhocrit_func(z)/(2*r200c_func(m200c, z))
        x = r/r200c_func(m200c, z)[None, ...]
        unitconv = (u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)
        frac_b = cosmopars['Omega_b']/cosmopars['Omega_m']* 2.0*(cosmopars['XH']+1.0)/(5.0*cosmopars['XH']+3.0)
        factorfront = unitconv*frac_b*p200c
        func = lambda p: self.Pth_over_Pdel(x, gamma=-0.3, alpha=1,
                                            P0 = self.PLmz(m200c, z, p['P0_A0'], p['P0_alpham'], p['P0_alphaz']),
                                            xc=self.PLmz(m200c, z, p['xc_A0'], p['xc_alpham'], p['xc_alphaz']),
                                            beta=self.PLmz(m200c, z, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)















# class rhoB12(BaseGNFW):
#     studies = ['B12']
#     params = {'rho0_A0': [4000., 10**2.6],
#                 'rho0_alpham': [0.29, 0],
#                 'rho0_alphaz': [-0.66, 0],
#                 'alpha_A0_k': [0.88, 1],
#                 'alpha_alpham_k': [-0.03, 0],
#                 'alpha_alphaz_k': [0.19, 0],
#                 'beta_A0_k': [3.83, 2.6],
#                 'beta_alpham_k': [0.04, 0],
#                 'beta_alphaz_k': [-0.025, 0],
#                 'xc_A0_k': [2, 0.6], # Not sure about this one for B12
#                 'xc_alpham_k': [0, 0],  # Maybe this and the following can be cropped?
#                 'xc_alphaz_k': [0, 0],
#                 'gamma_k': [-0.2, -0.2]}
    
#     def __init__(self, study):
#         self.checkstudy(study)
    
#     def rho1h(self, r, m200c, z, rhocrit_func, r200c_func, cosmopars):
#         r, m200c = r[:, None, None], m200c[:, None]
#         x = r/r200c_func(m200c, z)[None, ...]
#         frac_b = cosmopars['Omega_b']/cosmopars['Omega_m']
#         rhocs = rhocrit_func(z)
#         factorfront = frac_b*rhocs
#         func = lambda p: self.rho_over_rhodel(x, gamma=p['gamma_k'], 
#                                             alpha=self.GNFW_PPL(m200c, z, p['alpha_A0_k'], p['beta_alpham_k'], p['beta_alphaz_k']),
#                                             rho0=self.GNFW_PPL(m200c, z, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
#                                             xc=self.GNFW_PPL(m200c, z, p['xc_A0_k'], p['xc_alpham_k'], p['xc_alphaz_k']),
#                                             beta=self.GNFW_PPL(m200c, z, p['beta_A0_k'], p['beta_alpham_k'], p['beta_alphaz_k']))
#         return lambda p={}: factorfront*func(self.p0 | p)
    
# class twohalofromfile:
#     files = {'EmilyTNG': '/global/homes/c/cpopik/Git/Capybara/Data/twohalo_cmass_average.txt'}
    
#     def __init__(self, profname):
#         if profname in self.files.keys():
#             self.twohalofile = self.files[profname]
#         else:
#             raise NameError(f"Sample {profname} doesn't exist, choose from available samples: {list(self.files.keys())}")

#     def Pth2h(self, rs, m, z):
#         rsdata, pth2hdata = np.loadtxt(self.twohalofile, unpack=True)[0:2]
#         pth2h = np.interp(rs, rsdata, pth2hdata)[:, None, None]*np.ones((rs.size, m.size, z.size))
#         return lambda p={'A2h_t':1}: p['A2h_t']*pth2h