"""
Collections of radial halo profiles used to forward model SZ signals, specifically thermal pressure and gas density.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import Models.Studies as Studies
import Models.HaloModels as HaloModels



class BaseProfile:
    def units(self, prof, units):  # handles units of rho and pth for cosmo and cgs
        if prof=='pres': 
            if units=='cosmo': return u.Msun/u.Mpc/u.s**2
            elif units=='cgs': return u.g/u.cm/u.s**2
            elif units=='kpc': return u.Msun/u.kpc/u.s**2
        elif prof=='dens': 
            if units=='cosmo': return u.Msun/u.Mpc**3
            if units=='kpc': return u.Msun/u.kpc**3
            elif units=='cgs': return u.g/u.cm**3

    def setdim(self, rs, zs, logMs):  # Set proper dimensions of rs, zs, Ms
        rs = rs if np.array(rs, ndmin=1).ndim==3 else np.array(rs, ndmin=1)[:, None, None]
        zs = zs if np.array(zs, ndmin=1).ndim==2 else np.array(zs, ndmin=1)[:, None]
        logMs = logMs if np.array(logMs, ndmin=1).ndim==1 else np.array(logMs, ndmin=1)
        return rs, zs, logMs


class Nagai2007(BaseProfile, Studies.Nagai2007):  # Pressure Profile from GADGET-2 made hydro sims
    models = {
        'Run': ['Obs', 'CSF', 'NR'],  # Observed/cooling+SF sims/Non-rad sims
        'Sample':['Rel', 'Unrel'],  # relaxed or unrelaxed sample
    }
    params = {  # Table A1
        'alpha': {'Obs': {'Rel':1.3}, 'CSF': {'Rel':1.3, 'Unrel': 1.4}, 'NR': {'Rel':1.1, 'Unrel': 1.4}},  # slope at r~r_s
        'beta': {'Obs': {'Rel':4.3}, 'CSF': {'Rel':4.3, 'Unrel': 4.3}, 'NR': {'Rel':4.3, 'Unrel': 4.3}},  # slope at r>>r_s
        'gamma': {'Obs': {'Rel':0.7}, 'CSF': {'Rel':1.1, 'Unrel': 1.0}, 'NR': {'Rel':0.3, 'Unrel': 0.9}},  # slope at r<<r_s
        'P0': {'Obs': {'Rel':3.3}, 'CSF': {'Rel':3.3, 'Unrel': 2}, 'NR': {'Rel':38.0, 'Unrel': 3.0}},  # pressure amplitude
        'c500': {'Obs': {'Rel':1.8}, 'CSF': {'Rel':1.8, 'Unrel': 1.5}, 'NR': {'Rel':3.0, 'Unrel': 1.5}},  # concentration
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def P500(self, z, logM500c, units='cosmo'):  # Eq 3
        val = 1.45e-11*u.erg/u.cm**3 * (10**logM500c/1e15)**(2/3) * (self.H(z)/self.H0)**(8/3)
        return val.to(self.units('pres', units))

    def PGNFW(self, x, gamma, alpha, beta, P0):  # Eq A1
        return P0 / (x**gamma * (1+x**alpha)**((beta-gamma)/alpha))

    def Pressure(self, r, z, logM500c, units='cosmo'):
        self.require(list(self.models.keys()))
        r, z, logM500c = self.setdim(r, z, logM500c)  # set proper dimensions
        P500 = self.P500(z, logM500c, units)
        x = r*u.Mpc/(self.r500c(z, logM500c))
        PGNFW = lambda p: self.PGNFW(x=x*p['c500'], gamma=p['gamma'], alpha=p['alpha'], beta=p['beta'], P0=p['P0'])
        return lambda p={}: P500*PGNFW(self.p0 | p)



class Arnaud2010(BaseProfile, Studies.Arnaud2010):  # Pressure Profile fit to REXCESS cluseters with XMM-Newton data
    models = {'model': ['norm', 'ST', 'coolcore', 'disturbed'],}  # different best-fit parameter sets
    params = {  # Eq 12
        'alpha': {'norm': 1.0510, 'ST': 1.0620, 'coolcore': 1.2223, 'disturbed': 0.7736},  # intermediate slope
        'beta': {'norm': 5.4905, 'ST': 5.4807, 'coolcore': 5.49, 'disturbed': 5.49},  # outer slope
        'gamma': {'norm': 0.3081, 'ST': 0.3292, 'coolcore': 0.7736, 'disturbed': 0.3798},  # central slope
        'P0': {'norm': 8.403, 'ST': 8.130, 'coolcore': 3.249, 'disturbed': 3.202},  # units of h^(-3/2)
        'c500': {'norm': 1.177, 'ST': 1.156, 'coolcore': 1.128, 'disturbed': 1.083},  # R500/rs
        # fixed params
        'alpha_P': 0.12,  # mass dependence
        }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        self.h70 = self.h*100/70

    def P500(self, z, logM500c, units='cosmo'):  # Eq 5
        val = 1.65e-3 * (self.H(z)/self.H0)**(8/3) * (10**logM500c/(3e14/self.h70))**(2/3) *self.h70**2 *u.keV/u.cm**3
        return val.to(self.units('pres', units))
    
    def PGNFW(self, x, gamma, alpha, beta, P0, c500):  # Eq 11
        return P0 / ((x*c500)**gamma * (1+(x*c500)**alpha)**((beta-gamma)/alpha))

    def alphaPp(self, x, alpha_P):  # Eq 13
        return 0.10-(alpha_P+0.10)*(x/0.5)**3/(1+x/0.5)**3
    
    def mdep(self, x, logM500c, alphaPprime=False):  # mass dependence factor, Eq 13?
        alphaPp = lambda p: self.alphaPp(x, p) if alphaPprime else 0
        return lambda p: (10**logM500c/(3e14/self.h70))**(p['alpha_P']+alphaPp(p))
    
    def Pressure(self, r, z, logM500c, units='cosmo', alphaPprime=False):  # Eq 4/8/10/13
        r, z, logM500c = self.setdim(r, z, logM500c)  # set proper dimensions
        P500 = self.P500(z, logM500c, units)
        x = r*u.Mpc/(self.r500c(z, logM500c))
        PGNFW = lambda p: self.PGNFW(x, gamma=p['gamma'], alpha=p['alpha'], beta=p['beta'], P0=p['P0']*self.h70**(-3/2), c500=p['c500'])
        mdep = self.mdep(x, logM500c, alphaPprime)
        return lambda p={}: P500*mdep(self.p0 | p)*PGNFW(self.p0 | p)



class Battaglia2011(BaseProfile, Studies.Battaglia2011):  # Pressure Profile from GADGET-2 made hydro sims
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
    
    

class Battaglia2016(BaseProfile, Studies.Battaglia2016):  # Density Profile from GADGET-2 hydro sims
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





    


# In progress below here


class Planck2013(BaseProfile, Studies.Planck2013): # In progress
    models = {'cluster':['All', 'cool', 'noncool'],
              'fixedp': ['3','2','1','0']}
    params = {
        'P0': {'All':{'3':6.32,'2':6.82,'1':6.41,'0':5.78}, 'cool':11.82, 'noncool':4.72},
        'c500': {'All':{'3':1.01,'2':1.13,'1':1.81,'0':1.84}, 'cool':0.60, 'noncool':2.19},
        'gamma': {'All':{'3':0.31,'2':0.31,'1':0.31,'0':0.35}, 'cool':0.31, 'noncool':0.31},
        'alpha': {'All':{'3':1.05,'2':1.05,'1':1.33,'0':1.39}, 'cool':0.76, 'noncool':1.82},
        'beta': {'All':{'3':5.49,'2':5.17,'1':4.13,'0':4.05}, 'cool':6.58, 'noncool':3.62},
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
    def P500(self, z, logM500c, units='cosmo'):
        return Arnaud2010(H=self.H).P500(z, logM500c, units)
    
    def PGNFW(self, x, gamma, alpha, beta, P0, c500):
        return Arnaud2010().PGNFW(x, gamma, alpha, beta, P0, c500)

    def mdep(self, x, logM500c):
        return Arnaud2010().mdep(x, logM500c)({'alpha_P': 0.12})
    
    def Pressure(self, r, z, logM500c, units='cosmo'):
        A10P = Arnaud2010(H=self.H, r500c=self.r500c).Pressure(r, z, logM500c, units)
        return lambda p={}: A10P(self.p0 | p)




class Moser2021(BaseProfile, Studies.Moser2021):  # TODO in progress
    models={'model': ['GNFW', 'GNFW1h'],  # with or without two halo term
            'match': ['matched', 'unmatched'],  # match cmass mass distribution or not
            'select': ['star', 'halo'],  # stellar or halo mass selected
            'samp': ['tot', 'red'],  # color selected for cmass
            'fit': ['2D', '3D'],  # fit to 3D profiles or 2D projected profiles
            } 
    params={
    # best-fit, Table 3 & 4, I know this is ungodly but blame Emily
    'logrho0': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 3.67, 'red': 3.28}, 'halo': {'tot': 4.34, 'red': 3.38}}, 'unmatched': {'star': {'tot': 4.11, 'red': 3.38}, 'halo': {'tot': 5.68, 'red': 3.48}}}, 'GNFW1h': {'matched': {'star': {'tot': 6.00, 'red': 6.00}, 'halo': {'tot': 6.00, 'red': 6.00}}, 'unmatched': {'star': {'tot': 6.00, 'red': 6.00}, 'halo': {'tot': 6.00, 'red': 6.00}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 3.67, 'red': 3.28}, 'halo': {'tot': 4.34, 'red': 3.39}}, 'unmatched': {'star': {'tot': 3.69, 'red': 3.29}, 'halo': {'tot': 4.42, 'red': 3.41}}}, 'GNFW1h': {'matched': {'star': {'tot': 3.47, 'red': 3.12}, 'halo': {'tot': 3.94, 'red': 3.13}}, 'unmatched': {'star': {'tot': 3.49, 'red': 3.14}, 'halo': {'tot': 4.03, 'red': 3.16}}}}},
    'beta_k': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 3.37, 'red': 3.20}, 'halo': {'tot': 3.44, 'red': 3.30}}, 'unmatched': {'star': {'tot': 3.30, 'red': 2.91}, 'halo': {'tot': 3.54, 'red': 3.03}}}, 'GNFW1h': {'matched': {'star': {'tot': 2.06, 'red': 1.99}, 'halo': {'tot': 2.03, 'red': 1.92}}, 'unmatched': {'star': {'tot': 1.97, 'red': 1.88}, 'halo': {'tot': 1.94, 'red': 1.84}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 3.37, 'red': 3.20}, 'halo': {'tot': 3.45, 'red': 3.30}}, 'unmatched': {'star': {'tot': 3.12, 'red': 2.95}, 'halo': {'tot': 3.13, 'red': 3.03}}}, 'GNFW1h': {'matched': {'star': {'tot': 2.86, 'red': 2.71}, 'halo': {'tot': 2.79, 'red': 2.62}}, 'unmatched': {'star': {'tot': 2.67, 'red': 2.52}, 'halo': {'tot': 2.56, 'red': 2.43}}}}},
    'alpha_k': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 0.64, 'red': 0.80}, 'halo': {'tot': 0.43, 'red': 0.68}}, 'unmatched': {'star': {'tot': 0.49, 'red': 0.68}, 'halo': {'tot': 0.30, 'red': 0.61}}}, 'GNFW1h': {'matched': {'star': {'tot': 0.17, 'red': 0.16}, 'halo': {'tot': 0.17, 'red': 0.16}}, 'unmatched': {'star': {'tot': 0.16, 'red': 0.16}, 'halo': {'tot': 0.16, 'red': 0.15}}}}},
    'A2h_k': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 1.40, 'red': 1.37}, 'halo': {'tot': 1.31, 'red': 1.34}}, 'unmatched': {'star': {'tot': 1.31, 'red': 1.28}, 'halo': {'tot': 1.20, 'red': 1.30}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 1.41, 'red': 1.37}, 'halo': {'tot': 1.32, 'red': 1.35}}, 'unmatched': {'star': {'tot': 1.38, 'red': 1.34}, 'halo': {'tot': 1.29, 'red': 1.33}}}}},
    'P0': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 4.18, 'red': 4.03}, 'halo': {'tot': 2.74, 'red': 2.86}}, 'unmatched': {'star': {'tot': 3.95, 'red': 4.91}, 'halo': {'tot': 3.84, 'red': 4.47}}}, 'GNFW1h': {'matched': {'star': {'tot': 18.33, 'red': 17.37}, 'halo': {'tot': 13.26, 'red': 13.80}}, 'unmatched': {'star': {'tot': 19.54, 'red': 19.23}, 'halo': {'tot': 16.64, 'red': 16.50}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 4.20, 'red': 4.05}, 'halo': {'tot': 2.78, 'red': 2.89}}, 'unmatched': {'star': {'tot': 6.80, 'red': 5.98}, 'halo': {'tot': 7.52, 'red': 4.86}}}, 'GNFW1h': {'matched': {'star': {'tot': 3.11, 'red': 3.08}, 'halo': {'tot': 1.79, 'red': 1.98}}, 'unmatched': {'star': {'tot': 5.12, 'red': 4.62}, 'halo': {'tot': 4.92, 'red': 3.39}}}}},
    'beta_t': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 6.09, 'red': 6.14}, 'halo': {'tot': 6.31, 'red': 5.80}}, 'unmatched': {'star': {'tot': 5.00, 'red': 4.65}, 'halo': {'tot': 4.61, 'red': 4.55}}}, 'GNFW1h': {'matched': {'star': {'tot': 1.53, 'red': 1.54}, 'halo': {'tot': 1.42, 'red': 1.41}}, 'unmatched': {'star': {'tot': 1.39, 'red': 1.42}, 'halo': {'tot': 1.29, 'red': 1.39}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 6.10, 'red': 6.16}, 'halo': {'tot': 6.36, 'red': 5.83}}, 'unmatched': {'star': {'tot': 4.88, 'red': 5.00}, 'halo': {'tot': 4.60, 'red': 4.66}}}, 'GNFW1h': {'matched': {'star': {'tot': 5.24, 'red': 5.35}, 'halo': {'tot': 5.07, 'red': 4.79}}, 'unmatched': {'star': {'tot': 4.26, 'red': 4.41}, 'halo': {'tot': 3.77, 'red': 3.90}}}}},
    'xc_t': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 0.97, 'red': 1.00}, 'halo': {'tot': 0.98, 'red': 0.93}}, 'unmatched': {'star': {'tot': 0.98, 'red': 0.98}, 'halo': {'tot': 0.98, 'red': 0.95}}}, 'GNFW1h': {'matched': {'star': {'tot': 0.06, 'red': 0.06}, 'halo': {'tot': 0.05, 'red': 0.05}}, 'unmatched': {'star': {'tot': 0.06, 'red': 0.08}, 'halo': {'tot': 0.07, 'red': 0.08}}}}},
    'A2h_t': {
        '3D': {'GNFW': {'matched': {'star': {'tot': 0.57, 'red': 0.53}, 'halo': {'tot': 0.51, 'red': 0.51}}, 'unmatched': {'star': {'tot': 0.49, 'red': 0.47}, 'halo': {'tot': 0.44, 'red': 0.47}}}},
        '2D': {'GNFW': {'matched': {'star': {'tot': 0.57, 'red': 0.53}, 'halo': {'tot': 0.52, 'red': 0.51}}, 'unmatched': {'star': {'tot': 0.55, 'red': 0.51}, 'halo': {'tot': 0.49, 'red': 0.49}}}}},

    # fixed parameters, Table 1
    'gamma_t': -0.3,  # 
    'gamma_k': -0.2,  # 
    'xc_k': 0.5,      # 
    'alpha_t': 1.0,
    'xc_t_2D': 2,  # TODO: check?
    'alpha_k_2D': 0.7,  # TODO: check?
    }
    
    logMs_2h = np.linspace(10, 15, 50)  # mass range for 2-halo term integration
    ks_2h = np.geomspace(1e-3, 1e3, 50, 10)  # k range for 2-halo term FFT

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
    # def pGNFW(self, x, rho0, xc, gamma, alpha, beta):  # GNFW 
    #     # NOTE: sign in exponent is different from paper, which has a typo, should be beta+gamma
    #     return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta+gamma)/alpha)
    #     else:  # as written, used in future studies
    #         return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

    def GNFW(self, x, rho0, xc, gamma, alpha, beta):  # GNFW used for density profile in B16
        # NOTE: tpo is included here
        return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

    def p1h_del(self, rs, zs, logMs):  # Eq 5
        self.require(['model', 'match', 'select', 'samp', 'r200c', 'fit'])
        rs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        xs = rs*u.Mpc/self.r200c(zs, logMs)
        if self.fit=='2D':
            rho_rhodel = lambda p: self.GNFW(xs, gamma=p['gamma_k'], xc=p['xc_k'], alpha=p['alpha_k_2D'], rho0=10**p['logrho0'], beta=p['beta_k'])
        else:
            rho_rhodel = lambda p: self.GNFW(xs, gamma=p['gamma_k'], xc=p['xc_k'], alpha=p['alpha_k'], rho0=10**p['logrho0'], beta=p['beta_k'])
        return lambda p={}: rho_rhodel(self.p0 | p) 

    def pc(self, zs, units='cosmo'):  # Eq 5
        return self.Fb*self.rhoc(zs).to(self.units('dens', units))  # prefactor and units

    def dens1h(self, rs, zs, logMs, units='cosmo'):  # right before eq A1
        pc, p_del = self.pc(zs, units), self.p1h_del(rs, zs, logMs)
        return lambda p={}: pc*p_del(self.p0 | p)

    def P1h_del(self, rs, zs, logMs):  # Eq 6
        self.require(['model', 'match', 'select', 'samp', 'r200c', 'fit'])
        rs, zs, logMs = self.setdim(rs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        xs = rs*u.Mpc/self.r200c(zs, logMs)
        if self.fit=='2D':
            Pth_Pdel = lambda p: Battaglia2011().MGNFW(xs, gamma=p['gamma_t'], alpha=p['alpha_t'], P0=p['P0'], xc=p['xc_t_2D'],beta=p['beta_t'])
        else:
            Pth_Pdel = lambda p: Battaglia2011().MGNFW(xs, gamma=p['gamma_t'], alpha=p['alpha_t'], P0=p['P0'], xc=p['xc_t'],beta=p['beta_t'])
        return lambda p={}: Pth_Pdel(self.p0 | p)

    def P200c(self, zs, logMs, units='cosmo'):  # Eq 6
        self.require(['rhoc', 'r200c'])  # require rhoc
        _, zs, logMs = self.setdim(1, zs, logMs)  # set proper dimensions
        p200c = c.G*(10**logMs*u.Msun)*200*self.rhoc(zs)/(2*self.r200c(zs, logMs))
        return self.Fb*p200c.to(self.units('pres', units))

    def pres1h(self, rs, zs, logMs, units='cosmo'):  # right before eq A1
        P200c, P_del = self.P200c(zs, logMs, units), self.P1h_del(rs, zs, logMs)
        return lambda p={}: P200c*P_del(self.p0 | p)
    
    def twohalo(self, rs, zs, logMs, logMs_2h):  # Eq 8
        self.require(['dndlogm', 'bh', 'Plin'])  # required functions
        fft = HaloModels.mcfit_package(rs=rs)  # setup FFT
        ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions
        ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)  # Assign proper dimensions [nr, nz, nm]

        windfunc = lambda k: np.where(k*0.7 > 1/50, 1, 0)  # two-halo window function, [k]=1/Mpc
        prefac = self.bh(zs, logMs)*self.Plin(ks, zs)*windfunc(ks)  # collect factors outside int
        intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
        P2h = lambda prof1h: prefac*(np.trapz(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
        return lambda prof1h: IFFT3D(P2h(prof1h)) *prof1h.unit  # IFFT to real space and return its units destroyed by the FFT

    def prof2h(self, rs, zs, logMs):  # linear two-halo calculation, Section II.C Eq 17
        lin2h = self.twohalo(rs, zs, logMs, self.logMs_2h)  # linear two-halo calculation
        return lambda prof, p={}: lin2h(prof(rs, zs, self.logMs_2h)(p))

    def dens2h(self, rs, zs, logMs, units='cosmo'):  # two-halo density component
        rho1h = Battaglia2016({'model':'AGN'}, rhoc=self.rhoc, r200c=self.r200c, **self.info).dens
        lin2hrho = self.prof2h(rs, zs, logMs)(rho1h).to(self.units('dens', units))
        # rho2h = lambda p={}: lin2hrho *p['A2h_k']
        # return lambda p={}: rho2h(self.p0 | p)
        return lambda p={}: lin2hrho

    def pres2h(self, rs, zs, logMs, units='cosmo'):  # two-halo pressure component
        lin2h = self.twohalo(rs, zs, logMs, self.logMs_2h)  # linear two-halo calculation
        pth1h = Battaglia2011(rhoc=self.rhoc, r200c=self.r200c, **self.info).pres
        lin2hPth = self.prof2h(rs, zs, logMs)(pth1h).to(self.units('pres', units))
        # Pth2h = lambda p={}: lin2hPth *p['A2h_t']
        # return lambda p={}: Pth2h(self.p0 | p)
        return lambda prof, p={}: lin2h(prof(rs, zs, self.logMs_2h)(p))

    def pres2h(self, rs, zs, logMs, units='cosmo'):  # two-halo pressure component
        lin2h = self.twohalo(rs, zs, logMs, self.logMs_2h)  # linear two-halo calculation
        pth1h = Battaglia2011(rhoc=self.rhoc, r200c=self.r200c, **self.info).pres
        ptypo = {'beta_A0': B16.p0['beta_A0']-2*B16.p0['gamma']}
        lin2hPth = self.prof2h(rs, zs, logMs)(pth1h).to(self.units('pres', units))
        # Pth2h = lambda p={}: lin2hPth *p['A2h_t']
        # return lambda p={}: Pth2h(self.p0 | p)
        return lambda prof, p={}: lin2h(prof(rs, zs, self.logMs_2h)(p))
    
    
    # def P2h_del(self, rs, zs, logMs):  # two-halo density component
    #     P200c, Pth_2h = self.P200c(zs, logMs), self.pres2h(rs, zs, logMs)
    #     return lambda p={}: Pth_2h(p)/P200c
    
    # def p2h_del(self, rs, zs, logMs):  # two-halo density component
    #     pc, rho_2h = self.pc(zs), self.dens2h(rs, zs, logMs)
    #     return lambda p={}: rho_2h(p)/pc


    def pres(self, rs, zs, logMs, units='cosmo'):
        if self.model=='GNFW1h':
            pres = self.pres1h(rs, zs, logMs, units)
        else:
            Pth_1h, Pth_2h = self.pres1h(rs, zs, logMs, units), self.pres2h(rs, zs, logMs, units)
            pres = lambda p={}: Pth_1h(p) + Pth_2h(p)*p['A2h_k']
        return lambda p={}: pres(self.p0 | p)

    def dens(self, rs, zs, logMs, units='cosmo'):
        if self.model=='GNFW1h':
            dens = self.dens1h(rs, zs, logMs, units)
        else:
            rho_1h, rho_2h = self.dens1h(rs, zs, logMs, units), self.dens2h(rs, zs, logMs, units)
            dens = lambda p={}: rho_1h(p) + rho_2h(p)*p['A2h_t']
        return lambda p={}: dens(self.p0 | p)






class Vikram2017(BaseProfile, Studies.Vikram2017):  # TODO in progress
    models = {}  # only one model
    params = { 
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
        # B11 = Battaglia2011(inputsdict | inputvars, **self.info)
        # self.P1h_del = B11.P_del
        # self.P1h = B11.P
        # self.P200c = B11.P200c

    def twohalo(self, rs, zs, logMs, logMs_2h):  # Eq 8
        self.require(['dndlogm', 'bh', 'Plin'])  # required functions
        
        fft = HaloModels.mcfit_package(rs=rs)  # setup FFT
        ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions
        ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)  # Assign proper dimensions [nr, nz, nm]

        prefac = self.bh(zs, logMs)*self.Plin(ks, zs)  # collect factors outside int
        intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
        P2h = lambda prof1h: prefac*(np.trapz(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
        return lambda prof1h: IFFT3D(P2h(prof1h)) *prof1h.unit  # IFFT to real space and return its units destroyed by the FFT
    
    # def twohalo(self, rs, zs, logMs, logMs_2h):  # Eq 8
    #     self.require(['dndlogm', 'bh', 'Plin'])  # required functions
        
    #     Npad=1
    #     dlogr = np.log(rs[1]/rs[0])
    #     rspad = rs[0] * np.exp(-dlogr * np.arange(Npad, 0, -1))
    #     print(rspad)
    #     rsnew = np.concatenate([rspad, rs])
    #     fft = HaloModels.mcfit_package(rs=rsnew)  # setup FFT
    #     ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions
    #     ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)  # Assign proper dimensions [nr, nz, nm]

    #     prefac = self.bh(zs, logMs)*self.Plin(ks, zs)  # collect factors outside int
    #     intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
    #     P2h = lambda prof1h: prefac*(np.trapz(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
    #     return lambda prof1hmod, p={}: IFFT3D(P2h(prof1hmod(rsnew, zs, logMs_2h)(p)))[Npad:] *prof1hmod(0, 0, 0)().unit  # IFFT to real space and return its units destroyed by the FFT



class Amodeo2021(BaseProfile, Studies.Amodeo2021):  # ACT DR5 y map and SDSS BOSS CMASS DR10, arxiv.org/abs/2009.05558 TODO: in progress
    models = {'model': ['GNFW', 'OBB'],}  # pres/dens profile model
    params = {
        # best-fit GNFW params, T2
        'logrho0': {'GNFW': 2.6},  # density log amplitude
        'xc_k': {'GNFW': 0.6},     # density core radius
        'beta_k': {'GNFW': 2.6},   # density outer slope
        'A2h_k': {'GNFW': 1.1},    # density 2h amplitude
        'P0': {'GNFW': 2.0},       # pressure amplitude
        'alpha_t': {'GNFW': 0.8},  # pressure intermediate slope
        'beta_t': {'GNFW': 2.6},   # pressure outer slope
        'A2h_t': {'GNFW': 0.7},    # pressure 2h amplitude
        # Fixed GNFW params
        'gamma_t': -0.3,  # fixed GNFW pres params, Section II.Cp3
        'xc_t_A0': 0.497, 'xc_t_alpham': -0.00865, 'xc_t_alphaz':0.731,
        'gamma_k': -0.2,  # fixed GNFW dens params, Section II.Cp2
        # 'alpha_k_A0': 0.88, 'alpha_k_alpham': -0.03, 'alpha_k_alphaz':0.19,
        'alpha_k':1,
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def pGNFW(self, x, rho0, xc, gamma, alpha, beta):  # Eq 16
        return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

    def pc(self, z, units='cosmo'):
        return self.Fb*self.rhoc(z).to(self.units('dens', units))  # prefactor and units
    
    def Density1h(self, r, z, logM200c, units='cosmo'):  # one-halo density component, II.C.eq16
        self.require(['model'])
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        pc = self.pc(z, units)
        x = r*u.Mpc/self.r200c(z, logM200c)
        pGNFW = lambda p: self.pGNFW(x, gamma=p['gamma_k'], alpha=p['alpha_k'], rho0=10**p['logrho0'], xc=p['xc_k'], beta=p['beta_k'])
        return lambda p={}: pc*pGNFW(self.p0 | p)

    def PGNFW(self, x, P0, xc, gamma, alpha, beta):  # Eq 17
        return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
    
    def P200c(self, z, logM200c, units='cosmo'): 
        P200c = c.G*(10**logM200c*u.Msun)*200*self.rhoc(z)/(2*self.r200c(z, logM200c))
        return self.Fb*P200c.to(self.units('pres', units))
    
    def Pressure1h(self, r, z, logM200c, units='cosmo'):  # one-halo density component, II.C.eq16
        self.require(['model'])
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        P200c = self.P200c(z, logM200c, units)
        x = r*u.Mpc/self.r200c(z, logM200c)
        xc = Battaglia2011().PL(z, logM200c, A0=self.p0['xc_t_A0'], alpham=self.p0['xc_t_alpham'], alphaz=self.p0['xc_t_alphaz'])
        PGNFW = lambda p: self.PGNFW(x, gamma=p['gamma_t'], alpha=p['alpha_t'], P0=p['P0'], xc=xc, beta=p['beta_t'])
        return lambda p={}: P200c*PGNFW(self.p0 | p)

    def prof2h(self, r, z, logM200c):  # linear two-halo calculation, Section II.C Eq 17
        dndlogm = lambda z, logM200c: self.dndlogm(z, logM200c+np.log10(self.h)) * self.h**4
        bh = lambda z, logM200c: self.bh(z, logM200c+np.log10(self.h))
        Plin = lambda k, z: np.where(k*self.h > 1/50, 1, 0) * self.Plin(k*self.h, z) / self.h**3
        
        # dndlogm = lambda z, logM200c: self.dndlogm(z, logM200c)
        # bh = lambda z, logM200c: self.bh(z, logM200c)
        # Plin = lambda k, z: np.where(k > 1/50, 1, 0) * self.Plin(k, z)
        
        V17 = Vikram2017(dndlogm=dndlogm, bh=bh, Plin=Plin)
        logM200c_2h = np.linspace(10-np.log10(self.h), 15-np.log10(self.h), 50)
        lin2h = V17.twohalo(r, z, logM200c, logM200c_2h)  # linear two-halo calculation
        return lambda prof, p={}: lin2h(prof(r, z, logM200c_2h)(p))

    def Density2h(self, r, z, logM200c, units='cosmo'):  # two-halo density component
        B16 = Battaglia2016({'model':'AGN'}, rhoc=self.rhoc, r200c=self.r200c, **self.info)
        ptypo = {'beta_A0': B16.p0['beta_A0']-2*B16.p0['gamma']}
        Density1h = lambda r, z, logM200c: lambda p: B16.Density(r, z, logM200c, units)(ptypo)
        return self.prof2h(r, z, logM200c)(Density1h).to(self.units('dens', units))

    def Pressure2h(self, r, z, logM200c, units='cosmo'):  # two-halo pressure component
        B11 = Battaglia2011(rhoc=self.rhoc, r200c=self.r200c, **self.info)
        return self.prof2h(r, z, logM200c)(B11.Pressure).to(self.units('pres', units))

    def Pressure(self, r, z, logM200c, units='cosmo'):
        P1h, P2h_lin = self.Pressure1h(r, z, logM200c, units), self.Pressure2h(r, z, logM200c, units)
        P2h = lambda p: p['A2h_t']*P2h_lin
        return lambda p={}: P1h(self.p0 | p) + P2h(self.p0 | p)
    
    def Density(self, r, z, logM200c, units='cosmo'):
        p1h, p2h_lin = self.Density1h(r, z, logM200c, units), self.Density2h(r, z, logM200c, units)
        p2h = lambda p: p['A2h_k']*p2h_lin
        return lambda p={}: p1h(self.p0 | p) + p2h(self.p0 | p)


class Popik2026(BaseProfile, Studies.Amodeo2021):  # 
    models = {}
    params = {
        # Density Parameters
        'logrho0': 2.6, 'logrho0_z': -0.66, 'logrho0_m': 0.29,  # LOG amplitude
        'xc_k': 0.6, 'xc_k_z': 0, 'xc_k_m': 0,   # Core radius
        'beta_k': 2.6, 'beta_k_z': -0.025, 'beta_k_m': 0.04,  # Outer slope
        'gamma_k': -0.2, 'gamma_k_z': 0, 'gamma_k_m': 0,  # Inner Slope
        'alpha_k': 1, 'alpha_k_z': 0.19, 'alpha_k_m': -0.03,  # Intermediate Slope
        'A2h_k': 1,    # 2h amplitude

        # Pressure Parameters
        'P0': 2.0, 'P0_z': -0.758, 'P0_m': 0.154,       # Amplitude
        'alpha_t': 0.8, 'alpha_t_z': 0, 'alpha_t_m': 0,  # Intermediate slope
        'beta_t': 2.6, 'beta_t_z': 0.415, 'beta_t_m': 0.0393,  # Outer slope
        'gamma_t': -0.3, 'gamma_t_z': 0, 'gamma_t_m': 0, # Inner Slope
        'xc_t': 0.497, 'xc_t_z': 0.731, 'xc_t_m': -0.00865,   # Core Radius
        'A2h_t': 1,   # 2h amplitude
    }
    PLparams = {'zPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t'],
              'mPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t']}  # pres/dens profile model
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def PL(self, z, logM200c, pname):
        if not hasattr(self, 'zPL'): setattr(self, 'zPL', [])
        if not hasattr(self, 'mPL'): setattr(self, 'mPL', [])
        zterm = lambda alphaz: (1+z)**alphaz
        mterm = lambda alpham: (10**logM200c/1e14)**alpham
        zterm0 = zterm(self.p0[f"{pname}_z"])
        mterm0 = mterm(self.p0[f"{pname}_m"])

        func = lambda A0: A0
        
        if pname in self.zPL: func1 = lambda A0, alphaz: func(A0=A0)*zterm(alphaz)
        else: func1 = lambda A0, alphaz: func(A0)*zterm0
        
        if pname in self.mPL: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm(alpham)
        else: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm0

        return func2

    def pGNFW(self, x, rho0, xc, gamma, alpha, beta):
        return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

    def pc(self, z, units='cosmo'):
        return self.Fb*self.rhoc(z).to(self.units('dens', units))  # prefactor and units

    def Density1h(self, r, z, logM200c, units='cosmo'): 
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        pc = self.pc(z, units)
        x = r*u.Mpc/self.r200c(z, logM200c)

        gamma=self.PL(z, logM200c, 'gamma_k')
        alpha=self.PL(z, logM200c, 'alpha_k') 
        rho0=self.PL(z, logM200c, 'logrho0')
        xc=self.PL(z, logM200c, 'xc_k')
        beta=self.PL(z, logM200c, 'beta_k')

        pGNFW = lambda p: self.pGNFW(x, 
            gamma=gamma(p['gamma_k'], p['gamma_k_z'], p['gamma_k_m']), 
            alpha=alpha(p['alpha_k'], p['alpha_k_z'], p['alpha_k_m']), 
            rho0=rho0(10**p['logrho0'], p['logrho0_z'], p['logrho0_m']), 
            xc=xc(p['xc_k'], p['xc_k_z'], p['xc_k_m']), 
            beta=beta(p['beta_k'], p['beta_k_z'], p['beta_k_m']))
        return lambda p={}: pc*pGNFW(self.p0 | p)

    def PGNFW(self, x, P0, xc, gamma, alpha, beta):  # Eq 17
        return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
    
    def P200c(self, z, logM200c, units='cosmo'): 
        P200c = c.G*(10**logM200c*u.Msun)*200*self.rhoc(z)/(2*self.r200c(z, logM200c))
        return self.Fb*P200c.to(self.units('pres', units))
    
    def Pressure1h(self, r, z, logM200c, units='cosmo'):  
        r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
        P200c = self.P200c(z, logM200c, units)
        x = r*u.Mpc/self.r200c(z, logM200c)
        
        gamma=self.PL(z, logM200c, 'gamma_t')
        alpha=self.PL(z, logM200c, 'alpha_t') 
        P0=self.PL(z, logM200c, 'P0')
        xc=self.PL(z, logM200c, 'xc_t')
        beta=self.PL(z, logM200c, 'beta_t')
        
        PGNFW = lambda p: self.PGNFW(x, 
            gamma=gamma(p['gamma_t'], p['gamma_t_z'], p['gamma_t_m']), 
            alpha=alpha(p['alpha_t'], p['alpha_t_z'], p['alpha_t_m']), 
            P0=P0(p['P0'], p['P0_z'], p['P0_m']), 
            xc=xc(p['xc_t'], p['xc_t_z'], p['xc_t_m']), 
            beta=beta(p['beta_t'], p['beta_t_z'], p['beta_t_m']))
        return lambda p={}: P200c*PGNFW(self.p0 | p)

    def prof2h(self, r, z, logM200c): 
        dndlogm = lambda z, logM200c: self.dndlogm(z, logM200c)
        bh = lambda z, logM200c: self.bh(z, logM200c)
        Plin = lambda k, z: self.Plin(k, z)
        
        V17 = Vikram2017(dndlogm=dndlogm, bh=bh, Plin=Plin)
        logM200c_2h = np.linspace(10, 15, 10)
        lin2h = V17.twohalo(r, z, logM200c, logM200c_2h)  # linear two-halo calculation
        return lambda prof, p={}: lin2h(prof(r, z, logM200c_2h)(p))

    def Density2h(self, r, z, logM200c, units='cosmo'):  # two-halo density component
        twohalocalc = self.prof2h(r, z, logM200c)
        return lambda p={}: (self.p0 | p)['A2h_k']*twohalocalc(self.Density1h, p | {par[:-3]: p[par] for par in p if '_2h' in par}).to(self.units('dens', units))

    def Pressure2h(self, r, z, logM200c, units='cosmo'):  # two-halo pressure component
        twohalocalc = self.prof2h(r, z, logM200c)
        return lambda p={}: (self.p0 | p)['A2h_t']*twohalocalc(self.Pressure1h, p ).to(self.units('pres', units))

    def Pressure(self, r, z, logM200c, units='cosmo'):
        P1h, P2h = self.Pressure1h(r, z, logM200c, units), self.Pressure2h(r, z, logM200c, units)
        return lambda p={}: P1h(self.p0 | p) + P2h(self.p0 | p)
    
    def Density(self, r, z, logM200c, units='cosmo'):
        p1h, p2h = self.Density1h(r, z, logM200c, units), self.Density2h(r, z, logM200c, units)
        return lambda p={}: p1h(self.p0 | p) + p2h(self.p0 | p)




class Chen2023(BaseProfile, Studies.Chen2023):  # https://arxiv.org/pdf/2201.12591
    models = {}
    params = {
        # fixed
        'mu_e':1.17,
        'mu_p':1.17,
        'gamma':1.17
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    # def Pdel(self, zs, logMs, units='cosmo'):  # Eq.7
    #     self.require(['rvir'])
    #     _, zs, logMs = self.setdim(1, zs, logMs)  # set proper dimensions [nr, nz, nM]
    #     Tv = 2*c.G*10**logMs*u.Msun*c.m_p*self.p0['mu_e']*(1+zs)/self.rvir(zs, logMs)/c.k_B/3
    #     return Tv*c.k_B

    def conc(self, zs, logMs):
        return 7.85*(10**logMs/2e12)**(-0.081)*(1+zs)**(-0.71)

    # def Pe_del(self, xs, zs, logMs):
    #     xs, zs, logMs = self.setdim(xs, zs, logMs)  # set proper dimensions [nr, nz, nM]
    #     rho_gas = (np.log(1+xs*self.conc(zs, logMs))/(xs*self.conc(zs, logMs)))**(1/(self.p0['gamma']-1))
    #     return rho_gas/c.m_p/self.p0['mu_e'] * u.Msun/u.Mpc**3
    
    def Pe_del(self, xs, zs, logMs):
        xs, zs, logMs = self.setdim(xs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        P_del = (np.log(1+xs*self.conc(zs, logMs))/(xs*self.conc(zs, logMs)))**(1/(self.p0['gamma']-1)+1)
        return P_del
    
    def Pdel(self, zs, logMs, units='cosmo'):  # Eq.7
        self.require(['rhoc', 'rvir'])
        _, zs, logMs = self.setdim(1, zs, logMs)  # set proper dimensions [nr, nz, nM]
        Tv = 2/c.k_B/3 *c.G*10**logMs*u.Msun*c.m_p*self.p0['mu_p']*(1+zs)/self.rvir(zs, logMs)
        return (Tv*c.k_B/c.m_p/self.p0['mu_e'] *1e2*self.rhoc(zs)).to(self.units('pres', units))
    
    
    def Pe1h(self, rs, zs, logMs, units='cosmo'):
        Pdel = self.Pdel(zs, logMs, units)
        Pe_del = self.Pe_del(rs/self.rvir(zs, logMs), zs, logMs)
        return lambda p={}: Pdel*Pe_del



class Kou2023(BaseProfile, Studies.Kou2023):  # Planck 2018 and SDSS BOSS CMASS DR12
    models = {'mbin':['M1', "M2", "M3", "M4"],}  # mass bin
    params = {
        # Best-fit parameters
        "bh_m1": {"M1": 0.602, "M2": 0.623, "M3": 0.558, "M4": 0.550},  # hydrostatic bias
        # Fixed parameters
        'alpha_p': 0.12,  # Eq32
        'P0':6.41, 'gamma':0.31, 'alpha':1.33, 'beta':4.13, 'c_Pe':1.81, # eq 48
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    def Pe_del(self, xs):
        xs, _, _ = self.setdim(xs, 1, 1)  # set proper dimensions [nr, nz, nM]
        gnfw = lambda x, c, alpha, beta, gamma, P0: P0/((x*c)**gamma * (1+(x*c)**alpha)**((beta-gamma)/alpha))
        return gnfw(xs, c=self.p0['c_Pe'], P0=self.p0['P0'], gamma=self.p0['gamma'], alpha=self.p0['alpha'], beta=self.p0['beta'])

    def Pdel(self, zs, logMs, units='cosmo'):
        self.require(['H'])
        prefac = (1.65*(self.h/0.7)**2 * u.eV/u.cm**3 * (self.H(zs)/(self.H0))**(8/3)).to(self.units('pres', units))
        infac = (10**logMs/(3e14*0.7/self.h))
        Pdel = lambda p: prefac * (infac*p['bh_m1'])**(2/3+self.p0['alpha_p'])
        return lambda p={}: Pdel(self.p0 | p)
    
    def Pe1h(self, rs, zs, logMs, units='cosmo'):
        self.require(['r200m'])
        Pdel = self.Pdel(zs, logMs, units)
        logMs_ = lambda p: np.log10(p['bh_m1'])+logMs
        Pe_del = lambda p: self.Pe_del(rs*u.Mpc/self.r200m(zs, logMs_(p)))
        return lambda p={}: Pdel(self.p0 | p)*Pe_del(self.p0 | p)