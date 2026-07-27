"""
Some Future Paper
"""

from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Popik2026")




"""Old implementation being phased out"""


from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # In progress
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],
    }
    info = {
        'name':'Popik 2025'
        }



class Popik2026(Study):  # TODO: In progress
    path = f"{DATA_PATH}/Results"
    subs = {
    'zbin': ['z1', 'z2', 'z3', 'z4'],
    'deproj' : ['Base', 'cib', 'cib_cibdBeta', 'cib_cibdBeta_cibdT', 'cib_cibdT'],
    'TCIB': ['10.7', '24.0'],
    'beta': ['1.0', '1.2', '1.4', '1.6', '1.7', '1.8', '2.0'],
    }

    def __init__(self, inputsdict, **inputvars):        
        self.setup(inputsdict | inputvars)
        
        self.require(['deproj'])
        if self.deproj!='cib_cibdBeta_cibdT': # Add values
            self.subs['TCIB'] = ['10.7']
        if self.deproj!='cib_cibdBeta': # Add values
            self.subs['TCIB'] = self.subs['TCIB']+['1.15', '1.30', '1.35', '1.45']

        if self.deproj=='Base': self.require(['deproj'])
        else: self.require(['deproj', 'TCIB', 'beta'])


        self.get_meas()

        # Taking properties from Zhou
        Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
        for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
            setattr(self, val, getattr(Zhou, val))

    def get_meas(self):
        with h5py.File(f"{self.path}/ACTDR6DESILRG_Spectra_testnew.h5", 'r') as f:
            self.ell = f['ell'][()]
            self.Cgg_data = f[f'gxg/{self.zbin[-1]}'][()]
            self.Cgy_data = f[f'gxy/{self.zbin[-1]}/{self.deproj}/{self.TCIB}/{self.beta}'][()]
            self.Cyy_data = f[f'yxy/{self.deproj}/{self.TCIB}/{self.beta}'][()]

            self.Cgy_err = np.abs(self.Cgy_data)/10000*self.ell  # TODO: placeholder
            
            
from Models.Profiles import BaseProfile
from Models.Papers import Amodeo2021
class HaloProfile(BaseProfile, Amodeo2021.Study):  #
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
    
    
    
from Models.TargetData import BaseTargetData
class Popik2026(BaseTargetData, Study):  # TODO: In progress
    path = f"{DATA_PATH}/Results"
    subs = {
    'zbin': ['z1', 'z2', 'z3', 'z4'],
    'deproj' : ['Base', 'cib', 'cib_cibdBeta', 'cib_cibdBeta_cibdT', 'cib_cibdT'],
    'TCIB': ['10.7', '24.0'],
    'beta': ['1.0', '1.2', '1.4', '1.6', '1.7', '1.8', '2.0'],
    }

    def __init__(self, inputsdict, **inputvars):        
        self.setup(inputsdict | inputvars)
        
        self.require(['deproj'])
        if self.deproj!='cib_cibdBeta_cibdT': # Add values
            self.subs['TCIB'] = ['10.7']
        if self.deproj!='cib_cibdBeta': # Add values
            self.subs['TCIB'] = self.subs['TCIB']+['1.15', '1.30', '1.35', '1.45']

        if self.deproj=='Base': self.require(['deproj'])
        else: self.require(['deproj', 'TCIB', 'beta'])

        ACTDR6 = Coulton2024()
        self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data


        # Taking properties from Zhou
        Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
        for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
            setattr(self, val, getattr(Zhou, val))