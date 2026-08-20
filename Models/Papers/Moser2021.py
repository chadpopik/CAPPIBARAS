"""
The Impacts of Modeling Choices on the Inference of Circumgalactic Medium Properties from Sunyaev-Zeldovich Observations

arxiv.org/pdf/2103.02469
ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Moser2021")


class Table3_bestfit(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_bestfit.csv"):
        super().__init__(filename)

class Table3_marginalized(ParamTable):
    def __init__(self, filename=f"{thispath}/table3_marginalized.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)

class Table4(ParamTable):  # best-fit values from 2D fits
    def __init__(self, filename=f"{thispath}/table4.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)
        

class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(7, 6),
             xlabel=r'$r/r_{200c}$', xlim=(8e-2, 6.2e0), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(5e-31, 1.2e-26), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(7, 6),
             xlabel=r'$r/r_{200c}$', xlim=(8e-2, 6.2e0), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1e-16, 1.6e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig3(BasePlots2):
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(7, 6),
             xlabel=r'$M_h \ [M_\odot]$', xlim=(10.8, 16), xscale='linear',
             ylabel=r'$M_s \ [M_\odot]$', ylim=(7, 12.5), yscale='linear'),
        dict(name='Fig3b', filename='Fig3b', figsize=(7, 6),
             xlabel=r'$\log_10(M^*)\ (M_\odot)$', xlim=(10.6, 11.8), xscale='linear',
             ylabel='', ylim=(2, 5.5e4), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig4row1(BasePlots2):
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(6.5e-31, 4e-26), yscale='log'),
        dict(name='Fig4b', filename='Fig4b', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1.5e-16, 1.1e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig6col1(BasePlots2):
    subplots = [[
        dict(name='Fig6a', filename='Fig6a', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$\rho_\text{gas} [\text{g cm}^{-3}]$', ylim=(6.5e-31, 4e-26), yscale='log'),
        dict(name='Fig6c', filename='Fig6c', figsize=(7, 6),
             xlabel=r'$R (\text{Mpc})$', xlim=(7.5e-3, 1.1e1), xscale='log',
             ylabel=r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$', ylim=(1.5e-16, 1.1e-11), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)




"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
from Models.Papers import Amodeo2021
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    subs = {}
    info = Amodeo2021.Study.info


from CAPPIBARAS.Models.OldModules.Profiles import BaseProfile
from Models.Papers import Battaglia2012, Battaglia2016
from Models import HaloModels
class Moser2021(BaseProfile, Study):  # TODO in progress
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
            Pth_Pdel = lambda p: Battaglia2012.HaloProfiles().MGNFW(xs, gamma=p['gamma_t'], alpha=p['alpha_t'], P0=p['P0'], xc=p['xc_t_2D'],beta=p['beta_t'])
        else:
            Pth_Pdel = lambda p: Battaglia2012.HaloProfiles().MGNFW(xs, gamma=p['gamma_t'], alpha=p['alpha_t'], P0=p['P0'], xc=p['xc_t'],beta=p['beta_t'])
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

        windfunc = lambda k: np.where(k*0.7 > 1/50, 1, 0)  # two-halo window function, [k]=1/Mpc
        prefac = self.bh(zs, logMs)*self.Plin(ks, zs)*windfunc(ks)  # collect factors outside int
        intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
        P2h = lambda prof1h: prefac*(np.trapezoid(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
        return lambda prof1h: IFFT3D(P2h(prof1h)) *prof1h.unit  # IFFT to real space and return its units destroyed by the FFT

    def prof2h(self, rs, zs, logMs):  # linear two-halo calculation, Section II.C Eq 17
        lin2h = self.twohalo(rs, zs, logMs, self.logMs_2h)  # linear two-halo calculation
        return lambda prof, p={}: lin2h(prof(rs, zs, self.logMs_2h)(p))

    def dens2h(self, rs, zs, logMs, units='cosmo'):  # two-halo density component
        rho1h = Battaglia2016.HaloProfiles({'model':'AGN'}, rhoc=self.rhoc, r200c=self.r200c, **self.info).Density
        lin2hrho = self.prof2h(rs, zs, logMs)(rho1h).to(self.units('dens', units))
        # rho2h = lambda p={}: lin2hrho *p['A2h_k']
        # return lambda p={}: rho2h(self.p0 | p)
        return lambda p={}: lin2hrho

    def pres2h(self, rs, zs, logMs, units='cosmo'):  # two-halo pressure component
        pth1h = Battaglia2012.HaloProfiles(rhoc=self.rhoc, r200c=self.r200c, **self.info).Pressure
        lin2hPth = self.prof2h(rs, zs, logMs)(pth1h).to(self.units('pres', units))
        # Pth2h = lambda p={}: lin2hPth *p['A2h_t']
        # return lambda p={}: Pth2h(self.p0 | p)
        return lambda p={}: lin2hPth

    # def pres2h(self, rs, zs, logMs, units='cosmo'):  # two-halo pressure component
    #     lin2h = self.twohalo(rs, zs, logMs, self.logMs_2h)  # linear two-halo calculation
    #     pth1h = Battaglia2012.HaloProfiles(rhoc=self.rhoc, r200c=self.r200c, **self.info).pres
    #     ptypo = {'beta_A0': B16.p0['beta_A0']-2*B16.p0['gamma']}
    #     lin2hPth = self.prof2h(rs, zs, logMs)(pth1h).to(self.units('pres', units))
    #     # Pth2h = lambda p={}: lin2hPth *p['A2h_t']
    #     # return lambda p={}: Pth2h(self.p0 | p)
    #     return lambda prof, p={}: lin2h(prof(rs, zs, self.logMs_2h)(p))
    
    
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