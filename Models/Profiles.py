"""
Collections of radial halo profiles used to forward model SZ signals, specifically thermal pressure and gas density.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c

class BaseGNFW:
    # Check validitity of model specifications and assign default parameters
    def checkspefs(self, spefs, required):
        for spef in required:
            if spefs[spef] in getattr(self, f"{spef}s"):
                setattr(self, spef, spefs[spef])
            else:
                raise NameError(f"{spef} {spefs[spef]} doesn't exist, choose from available {spef}s: {getattr(self, f'{spef}s')}")
        self.p0 = {param: self.params[param][self.models.index(self.model)] for param in self.params.keys()}
    
    # Form of Mh and z dependance of GNFW parameters in Battaglia 2012
    def PLmz(self, z, logm200c, A0, alpham, alphaz):
        return A0 * (10**logm200c/1.e14)**alpham * (1.+z)**alphaz

    # GNFW used for density profile
    def GNFW(self, x, rho0, xc, gamma, alpha, beta):
        return rho0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-(beta-gamma)/alpha)
    
    # Modified GNFW used for pressure profile
    def MGNFW(self, x, P0, xc, gamma, alpha, beta):
        return P0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)

    # Two-halo component calculated with linear theory
    def twohalo(self, logmhalo, Plin, bias, hmf, FFT_func, IFFT_func, windowfunc=1):
        prefac, intfac = bias*Plin[..., None]*windowfunc, hmf*bias  # Precalculate factors
        P2h = lambda prof1h: prefac*np.trapz(FFT_func(prof1h)*intfac,logmhalo)[..., None]
        return lambda prof1h: IFFT_func(P2h(prof1h))  # IFFT to real space



# BOSS DR10 cross-correlated with ACT DR5 (Amodeo+ 2021, arxiv.org/abs/2009.05558)
class Amodeo2021(BaseGNFW):
    info = {'Omega_m': 0.25, 'Omega_b': 0.044, 'Omega_L': 0.75, 'H0': 70, 'v_rms': 1.06e-3, 'X_H':0.76,
            'mdef': '200c',
            'mstar_mean': 3.3e13,  # mean stellar mass
            'mhalo_mean': 3e11,  # corresponding mean halo mass from SHMR
            'z_med': 0.55,  # median redshift of sample
            }

    models = ['GNFW']
    params = {'logrho0': [2.6],  # density log amplitude
              'xc_k': [0.6],  # core radius
              'beta_k': [2.6],  # outer slope
              'A2h_k': [1.1],  # density 2h amplitude
              'P0': [2.0],  # pressure amplitude
              'alpha_t': [0.8],  # intermediate slope
              'beta_t': [2.6],  # outer slope
              'A2h_t': [0.7],  # pressure 2h amplitude
            }
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['model'])

    def Pth1h(self, rs, zs, logms200c, rhocs, rs200c, Omega_b, Omega_m, **kwargs):
        rs, zs, rhocs = rs[:, None, None], zs[:, None], rhocs[:, None]  # Assign 3D dimensions
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value  # Put G into Halo units
        Ps200c = G_cosmo*(10**logms200c)*200*rhocs/(2*rs200c)  # Define scale pressure
        factorfront = (Omega_b/Omega_m)*Ps200c*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)
        xs = rs/rs200c[None, ...]

        # Assign parameters as done in the paper
        func = lambda p: self.MGNFW(xs, gamma=-0.3, alpha=p['alpha_t'], P0 = p['P0'], xc=self.PLmz(zs, logms200c, A0=0.497, alpham=-0.00865, alphaz=0.731), beta=p['beta_t'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def rho1h(self, rs, zs, logms200c, rhocs, rs200c, Omega_b, Omega_m, **kwargs):
        rs, zs, rhocs = rs[:, None, None], zs[:, None], rhocs[:, None]  # Assign proper dimensions on r and z
        factorfront = (Omega_b/Omega_m)*rhocs*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)
        xs = rs/rs200c[None, ...]

        # Assign parameters as done in the paper
        func = lambda p: self.GNFW(xs, gamma=-0.2, alpha=1, rho0=10**p['logrho0'], xc=p['xc_k'], beta=p['beta_k'])
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def Pth2h(self, rs, zs, logmshalo, rhocs, rs200c, Plin, bias, hmf, FFT_func, IFFT_func, ks, **kwargs):
        windfunc = np.array([1 if k>1/50 else 0 for k in ks])[:, None, None]  # twohalo Window function
        lin2h = self.twohalo(logmshalo, Plin, bias, hmf, FFT_func, IFFT_func, windowfunc=windfunc)
        prof1h = Battaglia2012({'model':'B12'}).Pth1h(rs, zs, logmshalo, rhocs, rs200c, **kwargs)  # Use B12 parameterization to get 2h term
        ave2h = np.trapz(lin2h(prof1h())*hmf)  # Uses average over halo mass
        twohalo = lambda p: p['A2h_t']*ave2h  # Multiplied by an amplitude factor
        return lambda p={}: twohalo(self.p0 | p)
    
    def rho2h(self, rs, zs, logmshalo, rhocs, rs200c, Plin, bias, hmf, FFT_func, IFFT_func, ks, **kwargs):
        windfunc = np.array([1 if k>1/50 else 0 for k in ks])[:, None, None]  # twohalo Window function
        lin2h = self.twohalo(logmshalo, Plin, bias, hmf, FFT_func, IFFT_func, windowfunc=windfunc)
        prof1h = Battaglia2015({'model':'AGN'}).rho1h(rs, zs, logmshalo, rhocs, rs200c, **kwargs)  # Use B15 parameterization to get 2h term
        ave2h = np.trapz(lin2h(prof1h())*hmf)  # Uses average over halo mass
        twohalo = lambda p: p['A2h_k']*ave2h  # Multiplied by an amplitude factor
        return lambda p={}: twohalo(self.p0 | p)
    


class Battaglia2015(BaseGNFW):  # Calibrated off SPH sims made from GADGET-2 (arxiv.org/abs/1607.02442)
    info = {'mdef': '200c'}

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
        
    def rho1h(self, rs, zs, logms200c, rhocs, rs200c, Omega_b, Omega_m, **kwargs):
        rs, zs, rhocs = rs[:, None, None], zs[:, None], rhocs[:, None]  # Assign proper dimensions of (nr, nz, nm)
        factorfront = Omega_b/Omega_m*rhocs*(u.Msun/u.Mpc**3).to(u.g/u.cm**3)  # Calculate prefactor once
        xs = rs/rs200c[None, ...]  # scaled radii
        
        # Assign parameters following the parameterization of the paper
        func = lambda p: self.GNFW(xs, gamma=-0.2, xc=0.5,
                                    alpha=self.PLmz(zs, logms200c, p['alpha_A0'], p['beta_alpham'], p['beta_alphaz']),
                                    rho0=self.PLmz(zs, logms200c, p['rho0_A0'], p['rho0_alpham'], p['rho0_alphaz']),
                                    beta=self.PLmz(zs, logms200c, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)

    def rho2h(self, rs, zs, logmshalo, rhocrit, r200c, Plin, bias, hmf, FFT_func, IFFT_func, **kwargs):
        lin2h = self.twohalo(logmshalo, Plin, bias, hmf, FFT_func, IFFT_func)
        prof1h = self.rho1h(rs, zs, logmshalo, rhocrit, r200c, **kwargs)
        return lambda p={}: lin2h(prof1h(self.p0 | p))



class Battaglia2012(BaseGNFW):  # SPH sims made from GADGET-2 (arxiv.org/abs/1109.3711)
    info = {'mdef': '200c'}
    
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

    def Pth1h(self, rs, zs, logms200c, rhocs, rs200c, Omega_b, Omega_m, **kwargs):
        rs, zs, rhocs = rs[:, None, None], zs[:, None], rhocs[:, None]  # Assign proper dimensions on r and z
        G_cosmo = c.G.to(u.Mpc**3/u.Msun/u.s**2).value  # Proper units of G
        p200c = G_cosmo*(10**logms200c)*200*rhocs/(2*rs200c)  # Scaled pressure of 200c sphere
        factorfront = (Omega_b/Omega_m)*p200c*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)  # calculate prefactor once
        xs = rs/rs200c[None, ...]

        # Assign parameters as done in the paper
        func = lambda p: self.MGNFW(xs, gamma=-0.3, alpha=1,
                                    P0=self.PLmz(zs, logms200c, p['P0_A0'], p['P0_alpham'], p['P0_alphaz']),
                                    xc=self.PLmz(zs, logms200c, p['xc_A0'], p['xc_alpham'], p['xc_alphaz']),
                                    beta=self.PLmz(zs, logms200c, p['beta_A0'], p['beta_alpham'], p['beta_alphaz']))
        return lambda p={}: factorfront*func(self.p0 | p)
    
    def Pth2h(self, rs, zs, logmshalo, rhoscrit, rs200c, Plin, bias, hmf, FFT_func, IFFT_func, **kwargs):
        lin2h = self.twohalo(logmshalo, Plin, bias, hmf, FFT_func, IFFT_func)
        prof1h = self.Pth1h(rs, zs, logmshalo, rhoscrit, rs200c, **kwargs)
        return lambda p={}: lin2h(prof1h(self.p0 | p))

    
