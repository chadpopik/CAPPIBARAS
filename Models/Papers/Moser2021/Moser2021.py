"""
The Impacts of Modeling Choices on the Inference of Circumgalactic Medium Properties from Sunyaev-Zeldovich Observations

arxiv.org/pdf/2103.02469
ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
"""

import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c


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


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    subs = {}
    info = Amodeo2021.info


class Profiles(BaseProfile, Studies.Moser2021):  # TODO in progress
    models={'model': ['GNFW', 'GNFW1h'],  # with or without two halo term
            'match': ['matched', 'unmatched'],  # match cmass mass distribution or not
            'select': ['star', 'halo'],  # stellar or halo mass selected
            'samp': ['tot', 'red'],  # color selected for cmass
            'fit': ['2D', '3D'],  # fit to 3D profiles or 2D projected profiles
            }
    # fixed parameters, Table 1
    fixedparams = {
        'gamma_t': -0.3,
        'gamma_k': -0.2,
        'xc_k': 0.5,
        'alpha_t': 1.0,
        'xc_t_2D': 2,  # TODO: check?
        'alpha_k_2D': 0.7,  # TODO: check?
    }

    logMs_2h = np.linspace(10, 15, 50)  # mass range for 2-halo term integration
    ks_2h = np.geomspace(1e-3, 1e3, 50, 10)  # k range for 2-halo term FFT

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)

        massselect = 'mstar' if self.select=='star' else 'mh'
        if self.fit=='3D':  # best-fit values, Table 3
            row = Table3_bestfit().getparams(massselect=massselect, colorselect=self.samp, halo=self.model, match=self.match).to_dict()
            row = {'logrho0': row['log10p0'], 'alpha_k': row['alphak'], 'beta_k': row['betak'], 'A2h_k': row['Ak2h'],
                   'P0': row['P0'], 'xc_t': row['xct'], 'beta_t': row['betat'], 'A2h_t': row['At2h']}
        else:  # best-fit values, Table 4
            row = Table4().getparams(massselect=massselect, colorselect=self.samp, halo=self.model, match=self.match).to_dict()
            row = {'logrho0': row['log10p0'], 'beta_k': row['betak'], 'A2h_k': row['Ak2h'],
                   'P0': row['P0'], 'beta_t': row['betat'], 'A2h_t': row['At2h']}

        self.p0 = self.fixedparams | {k: v for k, v in row.items() if pd.notna(v)}
        
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


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    def Fig2(self, width=14, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$r/r_{200c}$', r'$r/r_{200c}$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[8e-2,6.2e0],[8e-2,6.2e0]], ylim=[[5e-31,1.2e-26],[1e-16,1.6e-11]], xscale='log', yscale='log')

    def Fig3(self, width=14, height=6):
        return self.plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$M_h \ [M_\odot]$', r'$\log_10(M^*)\ (M_\odot)$'], ylabel=[r'$M_s \ [M_\odot]$', ''],
            xlim=[[10.8, 16],[10.6,11.8]], ylim=[[7,12.5],[2,5.5e4]], xscale=['linear','linear'], yscale=['linear','log'])

    def Fig4row1(self, width=14, height=6):
        return self.plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])

    def Fig6col1(self, width=14, height=6):
        return self.plot(filename=['Fig6a','Fig6c'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])
