"""

"""

import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator


# Get average profile with an HOD
def HODweighting(Nc, Ns, uck, usk, logM, zs, hmf, FFT3D, IFFT1D, dNdz, HODp=None, **kwargs):
    ngal = lambda p: np.trapz((Nc(p)+Ns(p))*hmf, logM)[:, None]
    Hg = lambda p: (Nc(p)*uck(p) + Ns(p)*usk(p))/ngal(p)  # Define the HOD cross-spectra function
    if HODp is not None:  # if HOD parameters aren't being fit, precalculate the HOD terms to save time
        Hg_norm = Hg(HODp)/np.trapz(np.trapz(Hg*hmf*dNdz, logM), zs)[..., None, None]  # normalized galaxy distribution
        infac = Hg_norm*hmf*dNdz[:, None]  # Precalculated integrand factor
        aveprof = lambda prof: np.trapz(np.trapz(FFT3D(prof)*infac, logM), zs)  # mass and redshift average
        return lambda prof: IFFT1D(aveprof(prof))

    infac = hmf*dNdz[:, None]
    Hg_norm = lambda p: Hg(p)/np.trapz(np.trapz(Hg(p)*infac, logM), zs)[:, None, None]  # normalized galaxy distribution
    aveprof = lambda prof, p: np.trapz(np.trapz(FFT3D(prof)*Hg_norm(p)*infac, logM), zs)  # mass average
    return lambda prof, p={}: IFFT1D(aveprof(prof, p))  # IFFT


class Kou2023:  # arxiv.org/abs/2211.07502
    def __init__(self):
        pass
    
    def SN(self, ells, area, dNdz, zs, **kwargs):  # Shot Noise
        frac = area/ (4*np.pi*(180/np.pi)**2)
        return 4*np.pi*frac/np.trapz(dNdz, zs) * np.ones(ells.shape)
    
    def C_ell(self, ells, ks, zs, W_A, W_B, chis, Hs, **kwargs):  # Spherical Harmonics
        # Check the ell arrays to be within the bounds set by the input ks
        ells_from_ks = ks[:, None]*chis
        if ells.min()<ells_from_ks.min(): raise ValueError(f"ell_min must be be above {ells_from_ks.min()}")
        elif ells.max()>ells_from_ks.max(): raise ValueError(f"ell_max must be below {ells_from_ks.max()}")
        ks_from_ells = (ells[:, None])/chis  # 2D array, [n_ells]/[n_zs] > [n_ells, n_zs]
        
        # Make 2D interpolator for k and z, and array of (k, z) points to put into it
        intp_points = np.stack((ks_from_ells, zs*np.ones(ks_from_ells.shape)), axis=-1)
        P_AB_int = lambda P_AB: RegularGridInterpolator((ks, zs), P_AB, bounds_error=False, fill_value=np.nan)(intp_points)

        intfac = W_A * W_B * Hs/c.c.to(u.km/u.s).value/chis**2  # Precalculate the simple part
        return lambda P_AB: np.trapz(intfac*P_AB_int(P_AB), zs)
    
    def W_g(self, dNdz, zs, **kwargs):  # Galaxy Kernel
        return dNdz/np.trapz(dNdz, zs)
    
    def W_y(self, zs, **kwargs):  # Compton y kernel
        return 1/(1+zs)
    
    def P1h(self, hmf, logM, **kwargs):  # Galaxy auto-spectra, one-halo
        return lambda Hx, Hy: np.trapz(Hx*Hy*hmf, logM)
    
    def P2h(self, hmf, logM, bh, Plin, **kwargs):  # Galaxy auto-spectra, two-halo
        # infac = bh*hmf*Plin[..., None]
        return lambda Hx, Hy: Plin*np.trapz(Hx*bh*hmf, logM)*np.trapz(Hy*bh*hmf, logM)
    
    def H_c(self, Nc, Ns, logM, hmf, **kwargs):  # Cross-spectra function for centrals
        return lambda p={}: Nc(p)/self.ngal(Nc, Ns, hmf, logM)(p)
    
    def H_s(self, Nc, Ns, usk, logM, hmf, **kwargs):    # Cross-spectra function for satellites
        return lambda p={}: Ns(p)*usk(p)/self.ngal(Nc, Ns, hmf, logM)(p)
    
    def ngal(self, Nc, Ns, hmf, logM, **kwargs):
        return lambda p={}: np.trapz((Nc(p)+Ns(p))*hmf, logM)[:, None]
    
    def H_y(self, FFT_func, Hz, XH, **kwargs):  # Cross-spectra function for Compton y
        # Precalculate
        efrac = (2+2*XH)/(3+5*XH)  # electron fraction
        yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value  # Conversion from P_e to y
        cgs_cosmo = (u.g/u.cm/u.s**2).to(u.M_sun/u.Mpc/u.s**2)  # Pressure in CGS to Msun/Mpc units
        infac = yfac * efrac * cgs_cosmo
        prefac = c.c.to(u.km/u.s).value/Hz[:, None]  # Hz in units of km/s/Mpc
        return lambda Pth: prefac*FFT_func(infac*Pth)
    
    def Pgg_1h(self, Nc, Ns, usk, logM, hmf, **kwargs):
        Hc = self.H_c(Nc, Ns, logM, hmf)
        Hs = self.H_s(Nc, Ns, usk, logM, hmf)
        P1h = self.P1h(hmf, logM)
        return lambda p={}: 2*P1h(Hc(p), Hs(p)) + P1h(Hs(p), Hs(p))

    def Pgg_2h(self, Nc, Ns, usk, logM, hmf, bh, Plin, **kwargs):
        Hc = self.H_c(Nc, Ns, logM, hmf)
        Hs = self.H_s(Nc, Ns, usk, logM, hmf)
        P2h = self.P2h(hmf, logM, bh, Plin)
        return lambda p={}: P2h(Hc(p), Hc(p)) + 2*P2h(Hc(p), Hs(p)) + P2h(Hs(p), Hs(p))

    def Pgy_1h(self, Nc, Ns, usk, logM, hmf, FFT_func, Hz, XH, **kwargs):
        Hc = self.H_c(Nc, Ns, logM, hmf)
        Hs = self.H_s(Nc, Ns, usk, logM, hmf)
        Hy = self.H_y(FFT_func, Hz, XH)
        P1h = self.P1h(hmf, logM)
        return lambda Pth, p={}: P1h(Hc(p), Hy(Pth)) + P1h(Hs(p), Hy(Pth))

    def Pgy_2h(self, Nc, Ns, usk, logM, hmf, FFT_func, Hz, XH, bh, Plin, **kwargs):
        Hc = self.H_c(Nc, Ns, logM, hmf)
        Hs = self.H_s(Nc, Ns, usk, logM, hmf)
        Hy = self.H_y(FFT_func, Hz, XH)
        P2h = self.P2h(hmf, logM, bh, Plin)
        return lambda Pth, p={}: P2h(Hc(p), Hy(Pth)) + P2h(Hs(p), Hy(Pth))
    
    def Cgg1h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, **kwargs):
        Pgg_1h = self.Pgg_1h(Nc, Ns, usk, logM, hmf)
        Cl = self.C_ell(ells, ks, zs, self.W_g(dNdz, zs), self.W_g(dNdz, zs), chis, Hs)
        return lambda p={}: Cl(Pgg_1h(p))
    
    def Cgg2h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, bh, Plin, **kwargs):
        Pgg_2h = self.Pgg_2h(Nc, Ns, usk, logM, hmf, bh, Plin)
        Cl = self.C_ell(ells, ks, zs, self.W_g(dNdz, zs), self.W_g(dNdz, zs), chis, Hs)
        return lambda p={}: Cl(Pgg_2h(p))
    
    def Cgy1h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, FFT_func, beam_ells, beam_data, XH, **kwargs):
        Pgy_1h = self.Pgy_1h(Nc, Ns, usk, logM, hmf, FFT_func, Hs, XH)
        Cl = self.C_ell(ells, ks, zs, self.W_g(dNdz, zs), self.W_y(zs), chis, Hs)
        beam = np.interp(ells, beam_ells, beam_data)
        return lambda Pth, p={}: beam*Cl(Pgy_1h(Pth, p))
    
    def Cgy2h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, FFT_func, bh, Plin, beam_ells, beam_data, XH, **kwargs):
        Pgy_2h = self.Pgy_2h(Nc, Ns, usk, logM, hmf, FFT_func, Hs, XH, bh, Plin)
        Cl = self.C_ell(ells, ks, zs, self.W_g(dNdz, zs), self.W_y(zs), chis, Hs)
        beam = np.interp(ells, beam_ells, beam_data)
        return lambda Pth, p={}: beam*Cl(Pgy_2h(Pth, p))


class Kusiak2023:
    def __init__(self):
        pass
    
    def uk_to_ul(self, ells, ks, chis, zs):  # Interpolate to ks that correspond to the same desired ells over all zs
        # Check the ell arrays to be within the bounds set by the input ks
        ells_from_ks = ks[:, None]*chis-1/2  # Define ells that correspond to the input ks
        if ells.min()<ells_from_ks.min(): raise ValueError(f"ell_min must be be above {ells_from_ks.min()}")  # Check input kmin is low enough for desired ellmin
        elif ells.max()>ells_from_ks.max(): raise ValueError(f"ell_max must be below {ells_from_ks.max()}")  # Check input kmax is low enough for desired ellmax
        ks_from_ells = (ells[:, None]+1/2)/chis  # Define ks that correpond to the desired ells, 2D array, [n_ells]/[n_zs] > [n_ells, n_zs]

        # Make 2D interpolator for k and z, and array of (k, z) points to put into it
        intp_points = np.stack((ks_from_ells, zs*np.ones(ks_from_ells.shape)), axis=-1)
        return lambda uk: RegularGridInterpolator((ks, zs), uk, bounds_error=False, fill_value=np.nan)(intp_points)

    def u_g(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, **kwargs):
        ul = lambda p: self.uk_to_ul(ells, ks, chis, zs)(usk(p))
        W_g = self.W_g(Hz, chis, dNdz, zs)
        ngal = self.ngal(Nc, Ns, hmf, logM)
        return lambda p={}: W_g / ngal(p) * (Nc(p)+Ns(p)*ul(p))
    
    def ngal(self, Nc, Ns, hmf, logM, **kwargs):
        return lambda p={}: np.trapz((Nc(p)+Ns(p))*hmf, logM)[:, None]
        
    def W_g(self, Hz, chis, dNdz, zs, **kwargs):
        phi_g = dNdz / np.trapz(dNdz, zs)
        return (Hz/c.c.to(u.km/u.s).value * phi_g/chis**2)[:, None]
    
    def Cgg_1h(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ells, ks, **kwargs):
        d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
        ug2 = self.ug2(Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs)
        return lambda p={}: np.trapz(d2V_dzdOmega*np.trapz(hmf*ug2(p), logM), zs)

    def ug2(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs, **kwargs):
        ul = lambda p: self.uk_to_ul(ells, ks, chis, zs)(usk(p))
        W_g = self.W_g(Hz, chis, dNdz, zs)
        ngal = self.ngal(Nc, Ns, hmf, logM)
        return lambda p={}: W_g**2/ngal(p)**2 * (Ns(p)**2*ul(p)**2 + 2*Ns(p)*ul(p))

    def Cgg_2h(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, Plin, bh, **kwargs):
        Plinl = self.uk_to_ul(ells, ks, chis, zs)(Plin)
        ug = self.u_g(ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks)
        d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
        return lambda p={}: np.trapz(d2V_dzdOmega*Plinl*np.trapz(bh*hmf*ug(p), logM)**2, zs)

    def SN(self, area, dNdz, ells, zs, **kwargs):
        return area*(u.deg**2).to(u.sr)/np.trapz(dNdz, zs) *np.ones(ells.shape)










# from hmvec
# def C_yy_new(self,ells,zs,ks,Ppp,gzs,dndz=None,zmin=None,zmax=None):
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     Wz1s = 1/(1+gzs)
#     Wz2s = 1/(1+gzs)
#     # Convert to y units
#     # 

# def C_gy_new(self,ells,zs,ks,Pgp,gzs,gdndz=None,zmin=None,zmax=None):
#     gzs = np.asarray(gzs)
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     nznorm = np.trapz(gdndz,gzs)
#     term = (c.sigma_T/(c.m_e*c.c**2)).to(u.s**2/u.M_sun)*u.M_sun/u.s**2
#     Wz1s = gdndz/nznorm
#     Wz2s = 1/(1+gzs)

#     return limber_integral(ells,zs,ks,Pgp,gzs,Wz1s,Wz2s,hzs,chis)

# def C_gg_new(self,ells,zs,ks,Pgg,gzs,gdndz=None,zmin=None,zmax=None):
#     gzs = np.asarray(gzs)
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     nznorm = np.trapz(gdndz,gzs)
#     Wz1s = gdndz/nznorm
#     Wz2s = gdndz/nznorm
#     return limber_integral(ells,zs,ks,Pgg,gzs,Wz1s,Wz2s,hzs,chis)


# def u_y(zs, mshalo, r200_func, dA_func):
#     l200c = dA_func(zs)[:, None]/r200_func(zs, mshalo)
#     prefac = 4*np.pi*r200_func(zs, mshalo)/l200c**2 * (c.sigma_T/c.m_e/c.c**2)
#     return lambda Pek: prefac*Pek

# def u_g(zs, mshalo, Nc, Ns, hmf):
#     ng = np.trapz((Nc(mshalo)+Ns(mshalo))*hmf(zs, mshalo), np.log10(mshalo))
    

# Limber Integral from hmvec
# def limber_integral2(ells,zs,ks,Pzks,gzs,Wz1s,Wz2s,hzs,chis):
#     """
#     Get C(ell) = \int dz (H(z)/c) W1(z) W2(z) Pzks(z,k=ell/chi) / chis**2.
#     ells: (nells,) multipoles looped over
#     zs: redshifts (npzs,) corresponding to Pzks
#     ks: comoving wavenumbers (nks,) corresponding to Pzks
#     Pzks: (npzs,nks) power specrum
#     gzs: (nzs,) corersponding to Wz1s, W2zs, Hzs and chis
#     Wz1s: weight function (nzs,)
#     Wz2s: weight function (nzs,)
#     hzs: Hubble parameter (nzs,) in *1/Mpc* (e.g. camb.results.h_of_z(z))
#     chis: comoving distances (nzs,)

#     We interpolate P(z,k)
#     """

#     hzs = np.array(hzs).reshape(-1)
#     Wz1s = np.array(Wz1s).reshape(-1)
#     Wz2s = np.array(Wz2s).reshape(-1)
#     chis = np.array(chis).reshape(-1)
    
#     prefactor = hzs * Wz1s * Wz2s   / chis**2.
#     zevals = gzs
#     if zs.size>1:            
#          f = interp2d(ks,zs,Pzks,bounds_error=True)     
#     else:      
#          f = interp1d(ks,Pzks[0],bounds_error=True)
#     Cells = np.zeros(ells.shape)
#     for i,ell in enumerate(ells):
#         kevals = (ell+0.5)/chis
#         if zs.size>1:
#             # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
#             # to get around scipy.interpolate limitations
#             interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zevals)[0]
#         else:
#             interpolated = f(kevals)
#         if zevals.size==1: Cells[i] = interpolated * prefactor
#         else: Cells[i] = np.trapz(interpolated*prefactor,zevals)
#     return Cells