"""
Collections of radial profiles (pressure/density so far) either from functional forms, constructed from an emulator, or loaded in from a data file, both one-halo and two-halo, for various studies and their fixed/inferred parameters.
Functions should return lambda functions which take in parameter dictionaries, so that parts of the calculation unreliant on sampled parameters don't have to be redone. These lambda functions should return 3D arrays over radius, halo mass (in m200c), and redshift, even if they don't use those mass/redshift arrays for a calculation, to ensure same dimensionality between one-halo and two-halo profiles.
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
    
    def PLmz(self, z, m200c, A0, alpham, alphaz):
        return A0 * (m200c/1.e14)**alpham * (1.+z)**alphaz
    
    def rho_over_rhodel(self, x, rho0, xc, gamma, alpha, beta):
        return rho0* (x/xc)**gamma * (1.+(x/xc)**alpha)**(-(beta-gamma)/alpha)
    
    def Pth_over_Pdel(self, x, P0, xc, gamma, alpha, beta):
        return P0* (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)


class Amodeo2021(BaseGNFW):  # BOSS DR10 cross-correlated with ACT DR5 (arxiv.org/abs/2009.05558)
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
    mdef = "200c"
    meanmass, medz = 3.3*10**13, 0.55
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])
        
        self.rs2hfile, self.rho2hfile, self.pth2hfile = np.genfromtxt(self.twohalofile, unpack=True)

    def Pth1h(self, rs, zs, ms200c, rhocrit_func, r200c_func, Omega_b, Omega_m, **kwargs):
        # Calculate factors that won't change with profile parameters
        rs, zs = rs[:, None, None], zs[:, None]  # Assign proper dimensions on r and z
        xs = rs/r200c_func(zs, ms200c)[None, ...]
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value
        frac_b = Omega_b/Omega_m
        Ps200c = G_cosmo*ms200c*200*rhocrit_func(zs)/(2*r200c_func(zs, ms200c))
        factorfront = frac_b*Ps200c*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)
                                
        # Assign parameters as done in the paper
        func = lambda p: self.Pth_over_Pdel(xs, gamma=-0.3, alpha=p['alpha_t'], P0 = p['P0'], xc=self.PLmz(zs, ms200c, 0.497, -0.00865, 0.731), beta=p['beta_t'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def rho1h(self, rs, zs, ms200c, rhocrit_func, r200c_func, Omega_b, Omega_m, **kwargs):
        # Calculate factors that won't change with profile parameters
        rs, zs = rs[:, None, None], zs[:, None]  # Assign proper dimensions on r and z
        xs = rs/r200c_func(zs, ms200c)[None, ...]
        frac_b = Omega_b/Omega_m
        factorfront = frac_b*rhocrit_func(zs)*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)
        
        # Assign parameters as done in the paper
        func = lambda p: self.rho_over_rhodel(xs, gamma=-0.2, alpha=1, rho0=10**p['logrho0'], xc=p['xc_k'], beta=p['beta_k'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def Pth2h(self, rs, zs, mshalo):  # TODO 1
        Pth2h = np.interp(rs, self.rs2hfile, self.pth2hfile)  # Interpolate to requested rs 
        Pth2h = Pth2h[:, None, None]*np.ones((rs.size, zs.size, mshalo.size))  # ensure proper dimension even without z/m dependence
        
        # Assign parameters
        func = lambda p: p['A2h_t']*Pth2h
        return lambda p={}: func(self.p0 | p)
    
    def rho2h(self, rs, zs, mshalo):  # TODO 1
        rho2h = np.interp(rs, self.rs2hfile, self.rho2hfile)  # Interpolate to requested rs 
        rho2h = rho2h[:, None, None]*np.ones((rs.size, zs.size, mshalo.size))  # ensure proper dimension even without z/m dependence
        func = lambda p: p['A2h_k']*rho2h
        return lambda p={}: func(self.p0 | p)


class Battaglia2015(BaseGNFW):  # SPH sims made from GADGET-2 (arxiv.org/abs/1607.02442)
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
    
    mdef = "200c"
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])
        
    def rho1h(self, rs, zs, ms200c, rhocrit_func, r200c_func, Omega_b, Omega_m, **kwargs):
        # Calculate factors that won't change with profile parameters
        rs, zs = rs[:, None, None], zs[:, None]  # Assign proper dimensions on r and z
        xs = rs/r200c_func(zs, ms200c)[None, ...]
        frac_b = Omega_b/Omega_m
        factorfront = frac_b*rhocrit_func(zs)*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)
        
        # Assign parameters
        func = lambda p: self.rho_over_rhodel(xs, gamma=-0.2, 
                                            alpha=self.PLmz(zs, ms200c, p['alpha_A0'], p['beta_alpham'], p['beta_alphaz']),
                                            rho0=self.PLmz(zs, ms200c, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
                                            xc=0.5,
                                            beta=self.PLmz(zs, ms200c, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)


class Battaglia2012(BaseGNFW):  # SPH sims made from GADGET-2 (arxiv.org/abs/1109.3711)
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
    
    mdef = "200c"
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])

    def Pth1h(self, rs, zs, ms200c, rhocrit_func, r200c_func, Omega_b, Omega_m, **kwargs):
        # Calculate factors that won't change with profile parameters
        rs, zs = rs[:, None, None], zs[:, None]  # Assign proper dimensions on r and z
        xs = rs/r200c_func(zs,ms200c)[None, ...]
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value
        p200c = G_cosmo*ms200c*200*rhocrit_func(zs)/(2*r200c_func(zs, ms200c))
        frac_b = Omega_b/Omega_m
        factorfront = frac_b*p200c*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)
                
        # Assign parameters as done in the paper
        func = lambda p: self.Pth_over_Pdel(xs, gamma=-0.3, alpha=1,
                                            P0 = self.PLmz(zs, ms200c, p['P0_A0'], p['P0_alpham'], p['P0_alphaz']),
                                            xc=self.PLmz(zs, ms200c, p['xc_A0'], p['xc_alpham'], p['xc_alphaz']),
                                            beta=self.PLmz(zs, ms200c, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)